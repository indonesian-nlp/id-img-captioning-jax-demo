import io
import uvicorn
from transformers import pipeline
from pydantic import BaseModel
from fastapi import FastAPI
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class Input(BaseModel):
    text: str
    model_name: str
    max_len: int
    temp: float
    top_k: int
    top_p: float
    do_sample: bool

class TextGenResponse(BaseModel):
    result: str

app = FastAPI(
    title="Indonesian GPT-2 demo",
    version="0.1.0",
)

model_small = GPT2LMHeadModel.from_pretrained("flax-community/gpt2-small-indonesian")
tokenizer_small = GPT2Tokenizer.from_pretrained("flax-community/gpt2-small-indonesian")

MODELS = {
    "GPT-2 Small": (
        model_small,
        tokenizer_small
    )
}

@app.get('/')
def get_root():
    return {'message': 'Indonesian GPT-2 demo'}

@app.post('/generate/', response_model=TextGenResponse)
def query_gpt(item: Input):
    model, tokenizer = MODELS[item.model_name]

    input_ids = tokenizer.encode(item.text, return_tensors='pt')
    output = model.generate(input_ids=input_ids,
                            max_length=item.max_len,
                            temperature=item.temp,
                            top_k=item.top_k,
                            top_p=item.top_p,
                            do_sample=item.do_sample)

    text = tokenizer.decode(output[0], 
                            skip_special_tokens=True)

    return {'result': text}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
