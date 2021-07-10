import json
import requests
from mtranslate import translate
from prompts import PROMPT_LIST
import streamlit as st
import random

backend = "http://fastapi:8000/generate"

def process(text: str, model_name: str, server_url: str):
    payload = {"text": text, "model_name": model_name}
    r = requests.post(
        server_url, data=json.dumps(payload), headers={"Content-Type": "application/json"}, timeout=8000
    )

    return json.loads(r.text)

st.title("Indonesian GPT-2")

st.markdown(
    """Indonesian GPT-2 demo. Part of the [Huggingface JAX/Flax event](https://discuss.huggingface.co/t/open-to-the-community-community-week-using-jax-flax-for-nlp-cv/)."""
)

option = st.selectbox('Model',(['GPT-2 Small']))

if option == "GPT-2 Small":
    model_name = "flax-community/gpt2-small-indonesian"

prompt = st.selectbox('Prompt',(list(PROMPT_LIST.keys())+["Custom"]))

if prompt == "Custom":
    prompt_box = "Enter your text here"
else:
    prompt_box = random.choice(PROMPT_LIST[prompt])

text = st.text_area("Enter text", prompt_box)

if st.button("Run"):
    with st.spinner(text="Getting results..."):
        st.subheader("Result")
        result = process(text=text, model_name=model_name, server_url=backend)
        st.write(result["result"])
        st.text("English translation")
        st.write(translate(result["result"], "en", "id"))
