import io
import uvicorn
from transformers import pipeline
from pydantic import BaseModel
from fastapi import FastAPI

class Input(BaseModel):
    text: str
    model_name: str

class TextGenResponse(BaseModel):
    result: str

app = FastAPI(
    title="Indonesian GPT-2 demo",
    version="0.1.0",
)

def get_model(model_name: str):
    nlp = pipeline(task='text-generation',
                   model=model_name)

    return nlp

@app.get('/')
def get_root():
    return {'message': 'Indonesian GPT-2 demo'}

@app.post('/generate/', response_model=TextGenResponse)
def query_gpt(item: Input):
    nlp = get_model(item.model_name)
    result = nlp(item.text)

    text = result[0]["generated_text"]

    return {'result': text}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
