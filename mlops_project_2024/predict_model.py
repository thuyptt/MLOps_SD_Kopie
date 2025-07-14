from fastapi import FastAPI, HTTPException,Form
from fastapi.responses import FileResponse
import torch
from diffusers import AutoPipelineForImage2Image
import requests
from PIL import Image
from io import BytesIO
import yaml
import wandb
from accelerate import Accelerator

app = FastAPI()

def load_config():
    config_path = "./mlops_project_2024/config/default_config.yaml"
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    return config

def download_lora_weights(config):
    try:
        wandb.init(project=config["PROJECT_NAME"], entity=config["ENTITY"], job_type="download_weights")
#        wandb.login(key=config["WANDB_API_KEY"])
        artifact = wandb.use_artifact(f"{config['ENTITY']}/{config['PROJECT_NAME']}/{config['ARTIFACT_NAME']}", type="model")
        artifact_dir = artifact.download(config['output_dir'])
        print(f"Weights downloaded to {artifact_dir}")
        return artifact_dir
    except Exception as e:
        print(f"Failed to download weights from wandb: {e}")
        return config.get('output_dir')

@app.on_event("startup")
async def startup_event():
    config = load_config()
    lora_weights_path = download_lora_weights(config)

    accelerator = Accelerator()
    device = accelerator.device
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    # Initialize the model pipeline
    global pipeline
    pipeline = AutoPipelineForImage2Image.from_pretrained(
        config["pretrained_model_name_or_path"],
        torch_dtype=dtype,
        use_safetensors=True
    ).to(device)
    pipeline.load_lora_weights(lora_weights_path, weight_name="pytorch_lora_weights.safetensors")


@app.post("/generate-image-file/", response_class=FileResponse)
async def generate_image_file(prompt: str = Form(...), image_url: str = Form(...)): # for form data request
# async def generate_image_file(prompt: str = Query(...), image_url: str = Query(...)): #for query string request
    try:
        # Load the image from URL
        response = requests.get(image_url)
        response.raise_for_status()
        init_image = Image.open(BytesIO(response.content))

        # Check if the image is quadratic
        if init_image.width != init_image.height:
            raise HTTPException(status_code=400, detail="The input image must be quadratic (equal width and height).")

        # Resize the image to 512x512
        init_image = init_image.resize((512, 512))

        # Generate the image using the predict function
        generated_image = pipeline(prompt, image = init_image, strength = 0.75, guidance_scale = 6.5, num_inference_steps = 30).images[0]

        # Save the image to a temporary file
        output_path = "generated_image.jpg"
        generated_image.save(output_path, format="JPEG")

        return FileResponse(output_path, media_type="image/jpeg", filename="generated_image.jpg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
