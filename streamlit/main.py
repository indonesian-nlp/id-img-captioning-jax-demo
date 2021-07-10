import json
import requests
from mtranslate import translate
from prompts import PROMPT_LIST
import streamlit as st
import random

backend = "http://fastapi:8000/generate"

def process(text: str,
            model_name: str,
            max_len: int,
            temp: float,
            top_k: int,
            top_p: float,
            do_sample: bool,
            server_url: str):
    payload = {"text": text,
               "max_len": max_len,
               "temp": temp,
               "top_k": top_k,
               "top_p": top_p,
               "do_sample": do_sample,
               "model_name": model_name}
    r = requests.post(
        server_url, data=json.dumps(payload), headers={"Content-Type": "application/json"}, timeout=8000
    )

    return json.loads(r.text)

st.title("Indonesian GPT-2")

st.sidebar.subheader("Configurable parameters")

max_len = st.sidebar.text_input(
    "Maximum length",
    value=100,
    help="The maximum length of the sequence to be generated."
)

temp = st.sidebar.slider(
    "Temperature",
    value=1.0,
    min_value=0.0,
    max_value=1.0,
    help="The value used to module the next token probabilities."
)

top_k = st.sidebar.text_input(
    "Top k",
    value=50,
    help="The number of highest probability vocabulary tokens to keep for top-k-filtering."
)

top_p = st.sidebar.text_input(
    "Top p",
    value=1.0,
    help=" If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation."
)

do_sample = st.sidebar.selectbox('Sampling?', (True, False), help="Whether or not to use sampling; use greedy decoding otherwise.")

st.markdown(
    """Indonesian GPT-2 demo. Part of the [Huggingface JAX/Flax event](https://discuss.huggingface.co/t/open-to-the-community-community-week-using-jax-flax-for-nlp-cv/)."""
)

model_name = st.selectbox('Model',(['GPT-2 Small']))

ALL_PROMPTS = list(PROMPT_LIST.keys())+["Custom"]
prompt = st.selectbox('Prompt', ALL_PROMPTS, index=len(ALL_PROMPTS)-1)

if prompt == "Custom":
    prompt_box = "Enter your text here"
else:
    prompt_box = random.choice(PROMPT_LIST[prompt])

text = st.text_area("Enter text", prompt_box)

if st.button("Run"):
    with st.spinner(text="Getting results..."):
        st.subheader("Result")
        result = process(text=text,
                         model_name=model_name,
                         max_len=max_len,
                         temp=temp,
                         top_k=top_k,
                         top_p=top_p,
                         do_sample=do_sample,
                         server_url=backend)
        st.write(result["result"])
        st.text("English translation")
        st.write(translate(result["result"], "en", "id"))
