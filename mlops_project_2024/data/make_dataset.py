#!/usr/bin/env python
# make_dataset.py
import os
import random
import numpy as np
from datasets import load_dataset
from transformers import CLIPTokenizer
import hydra
from omegaconf import DictConfig

def tokenize_captions(examples, tokenizer, caption_column, is_train=True):
    captions = []
    for caption in examples[caption_column]:
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            captions.append(random.choice(caption) if is_train else caption[0])
        else:
            raise ValueError(f"Caption column `{caption_column}` should contain either strings or lists of strings.")
    inputs = tokenizer(captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
    return inputs.input_ids

@hydra.main(config_path="./config", config_name="default_config_wo_dataset.yaml")
def main(cfg: DictConfig):

    # Load tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="tokenizer", revision=cfg.revision)

    # Get the datasets
    data_files = {"train": os.path.join(cfg.train_data_dir, "**")}
    dataset = load_dataset("imagefolder", data_files=data_files, cache_dir=cfg.cache_dir)

    # Preprocessing the datasets
    caption_column = cfg.caption_column

    def preprocess_train(examples):
        examples["input_ids"] = tokenize_captions(examples, tokenizer, caption_column)
        return examples

    # Preprocess and save the dataset
    train_dataset = dataset["train"].map(
        preprocess_train,
        batched=True,
        remove_columns=[caption_column]
    )
    train_dataset.save_to_disk(cfg.processed_dataset_path)

if __name__ == "__main__":
    main()
