#!/usr/bin/env python

# Custom Verions of the Huggingface's train_text_to_image_lora.py script

import hydra
from omegaconf import OmegaConf
import math
import os
import shutil
from contextlib import nullcontext
from pathlib import Path

import datasets
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import  load_from_disk
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from torchvision import transforms

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, compute_snr
from diffusers.utils import convert_state_dict_to_diffusers
from diffusers.utils.torch_utils import is_compiled_module
import wandb


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.30.0.dev0")

logger = get_logger(__name__, log_level="INFO")

def log_validation(pipeline, cfg, accelerator, epoch, is_final_validation=False):
    logger.info(
        f"Running validation... \n Generating {cfg.num_validation_images} images with prompt:"
        f" {cfg.validation_prompt}."
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    generator = torch.Generator(device=accelerator.device)
    if cfg.seed is not None:
        generator = generator.manual_seed(cfg.seed)
    images = []
    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)

    with autocast_ctx:
        for _ in range(cfg.num_validation_images):
            images.append(pipeline(cfg.validation_prompt, num_inference_steps=50, generator=generator).images[0])

    phase_name = "test" if is_final_validation else "validation"
    wandb.log(
        {
            phase_name: [
                wandb.Image(image, caption=f"{i}: {cfg.validation_prompt}") for i, image in enumerate(images)
            ]
        }
    )
    return images

@hydra.main(config_path="./config", config_name="default_config.yaml", version_base=None)
def main(cfg):

    logging_dir = Path(cfg.output_dir, cfg.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=cfg.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        project_config=accelerator_project_config,
    )

    os.path.abspath(__file__)

    wandb.init(
        project = "mlops2024-sd-lora",
        config = OmegaConf.to_container(cfg, resolve=True),
        entity = "banjuessing-LMU Munich",
        job_type = "training",
    #    program_relpath=script_path
    )

    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if cfg.seed is not None:
        set_seed(cfg.seed)

    if accelerator.is_main_process:
        if cfg.output_dir is not None:
            os.makedirs(cfg.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="scheduler")
    CLIPTokenizer.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="tokenizer", revision=cfg.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="text_encoder", revision=cfg.revision
    )
    vae = AutoencoderKL.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="vae", revision=cfg.revision, variant=cfg.variant
    )
    unet = UNet2DConditionModel.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="unet", revision=cfg.revision, variant=cfg.variant
    )
    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Freeze the unet parameters before adding adapters
    for param in unet.parameters():
        param.requires_grad_(False)

    unet_lora_config = LoraConfig(
        r=cfg.rank,
        lora_alpha=cfg.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # Add adapter and make sure the trainable params are in float32.
    unet.add_adapter(unet_lora_config)
    if cfg.mixed_precision == "fp16":
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(unet, dtype=torch.float32)

    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())

    # Enable TF32 for faster training on Ampere GPUs if necessary
    if cfg.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if cfg.scale_lr:
        cfg.learning_rate = (
            cfg.learning_rate * cfg.gradient_accumulation_steps * cfg.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        lora_layers,
        lr=cfg.learning_rate,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.adam_weight_decay,
        eps=cfg.adam_epsilon,
    )

    train_dataset = load_from_disk(cfg.processed_dataset_path)

    image_column = cfg.image_column

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model


    # Define transforms
    train_transforms = transforms.Compose([
        transforms.Resize(cfg.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(cfg.resolution) if cfg.center_crop else transforms.RandomCrop(cfg.resolution),
        transforms.RandomHorizontalFlip() if cfg.random_flip else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        return examples

    with accelerator.main_process_first():
        train_dataset = train_dataset.with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([torch.tensor(example["input_ids"]) for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=cfg.train_batch_size,
        num_workers=cfg.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    num_warmup_steps_for_scheduler = cfg.lr_warmup_steps * accelerator.num_processes
    if cfg.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / cfg.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            cfg.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = cfg.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
    if cfg.max_train_steps is None:
        cfg.max_train_steps = cfg.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != cfg.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    cfg.num_train_epochs = math.ceil(cfg.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = cfg.train_batch_size * accelerator.num_processes * cfg.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {cfg.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {cfg.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {cfg.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if cfg.resume_from_checkpoint:
        if cfg.resume_from_checkpoint != "latest":
            path = os.path.basename(cfg.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(cfg.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(f"Checkpoint '{cfg.resume_from_checkpoint}' does not exist. Starting a new training run.")
            cfg.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(cfg.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, cfg.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, cfg.num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if cfg.noise_offset:
                    noise += cfg.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )

                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]

                # Get the target for loss depending on the prediction type
                if cfg.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=cfg.prediction_type)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict the noise residual and compute loss
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]

                if cfg.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, cfg.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                        dim=1
                    )[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(cfg.train_batch_size)).mean()
                train_loss += avg_loss.item() / cfg.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = lora_layers
                    accelerator.clip_grad_norm_(params_to_clip, cfg.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                wandb.log({"train_loss": train_loss})
                train_loss = 0.0

                if global_step % cfg.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if cfg.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(cfg.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= cfg.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - cfg.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(cfg.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(cfg.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)

                        unwrapped_unet = unwrap_model(unet)
                        unet_lora_state_dict = convert_state_dict_to_diffusers(
                            get_peft_model_state_dict(unwrapped_unet)
                        )

                        StableDiffusionPipeline.save_lora_weights(
                            save_directory=save_path,
                            unet_lora_layers=unet_lora_state_dict,
                            safe_serialization=True,
                        )

                        logger.info(f"Saved state to {save_path}")
#                        wandb.save(os.path.join(save_path, '*'), base_path=save_path)

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= cfg.max_train_steps:
                break

        if accelerator.is_main_process:
            if cfg.validation_prompt is not None and epoch % cfg.validation_epochs == 0:
                # create pipeline
                pipeline = DiffusionPipeline.from_pretrained(
                    cfg.pretrained_model_name_or_path,
                    unet=unwrap_model(unet),
                    revision=cfg.revision,
                    variant=cfg.variant,
                    torch_dtype=weight_dtype,
                )
                images = log_validation(pipeline, cfg, accelerator, epoch)

                del pipeline
                torch.cuda.empty_cache()

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unet.to(torch.float32)

        unwrapped_unet = unwrap_model(unet)
        unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))
        StableDiffusionPipeline.save_lora_weights(
            save_directory=cfg.output_dir,
            unet_lora_layers=unet_lora_state_dict,
            safe_serialization=True,
        )
        artifact = wandb.Artifact('lora_weights', type='model')
        artifact.add_dir(cfg.output_dir)
        wandb.log_artifact(artifact)

        # Final inference
        # Load previous pipeline
        if cfg.validation_prompt is not None:
            pipeline = DiffusionPipeline.from_pretrained(
                cfg.pretrained_model_name_or_path,
                revision=cfg.revision,
                variant=cfg.variant,
                torch_dtype=weight_dtype,
            )

            # load attention processors
            pipeline.load_lora_weights(cfg.output_dir)

            # run inference
            images = log_validation(pipeline, cfg, accelerator, epoch, is_final_validation=True)
            wandb.log({"final_validation_images": [wandb.Image(img) for img in images]})

    wandb.finish()
    logger.info("WandB run finished.")


if __name__ == "__main__":
    main()
    exit(0)
