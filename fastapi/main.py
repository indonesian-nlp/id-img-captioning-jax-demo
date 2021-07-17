import io
import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI, File
import torch
from PIL import Image
from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize, ToTensor, Compose
from torchvision.transforms.functional import InterpolationMode
import torch
import numpy as np
from transformers import MarianTokenizer
from flax_clip_vision_marian.modeling_clip_vision_marian import FlaxCLIPVisionMarianForConditionalGeneration
import logging

class TextGenResponse(BaseModel):
    result: str

app = FastAPI(
    title="Indonesian Image Captioning demo",
    version="0.1.0",
)

logging.info("Loading tokenizer...")
marian_model_name = 'Helsinki-NLP/opus-mt-en-id'
tokenizer = MarianTokenizer.from_pretrained(marian_model_name)
logging.info("Tokenizer loaded.")

logging.info("Loading model...")
clip_marian_model_name = 'flax-community/Image-captioning-Indonesia'
model = FlaxCLIPVisionMarianForConditionalGeneration.from_pretrained(clip_marian_model_name)
logging.info("Model loaded.")

config = model.config
image_size = config.clip_vision_config.image_size
custom_transforms = torch.nn.Sequential(
                    Resize([image_size], interpolation=InterpolationMode.BICUBIC),
                    CenterCrop(image_size),
                    ConvertImageDtype(torch.float),
                    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                )

def generate_step(pixel_values, max_len, num_beams):
    gen_kwargs = {"max_length": int(max_len) , "num_beams": int(num_beams)}

    logging.info("Generating caption...")
    output_ids = model.generate(pixel_values, **gen_kwargs)
    token_ids = np.array(output_ids.sequences)[0]
    caption = tokenizer.decode(token_ids)
    logging.info("Caption generated.")

    return caption

@app.get('/')
def get_root():
    return {'message': 'Indonesian GPT-2 demo'}

def load_image(file):
    logging.info("Loading image...")
    input_image = Image.open(io.BytesIO(file)).convert("RGB")
    loader = Compose([ToTensor()])  
    image = loader(input_image)
    image = custom_transforms(image)
    pixel_values = torch.stack([image]).permute(0, 2, 3, 1).numpy()
    logging.info("Image loaded.")

    return pixel_values

@app.post('/generate', response_model=TextGenResponse)
def query_caption(file: bytes = File(...), max_len:int=8, num_beams: int=4):
    pixel_values = load_image(file)

    generated_ids = generate_step(pixel_values, max_len, num_beams)
    return {'result': generated_ids}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
