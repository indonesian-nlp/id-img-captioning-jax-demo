import io

import requests
from PIL import Image
from requests_toolbelt.multipart.encoder import MultipartEncoder
import json
from mtranslate import translate

import streamlit as st

backend = "http://fastapi:8000/generate"

def process(image, server_url: str, max_len:int, num_beams:int):
    m = MultipartEncoder(fields={"file": ("filename", image, "image/jpeg"), 
                                 "max_len": str(max_len),
                                 "num_beams": str(num_beams)})
    r = requests.post(
        server_url, data=m, headers={"Content-Type": m.content_type}, timeout=8000
    )

    return json.loads(r.text)

st.set_page_config(page_title="Indonesian Image Captioning Demo", page_icon="🖼️")

st.title("Indonesian Image Captioning Demo")

st.markdown(
    """Indonesian image captioning demo, trained on [CLIP](https://huggingface.co/transformers/model_doc/clip.html) and [Marian](https://huggingface.co/transformers/model_doc/marian.html). Part of the [Huggingface JAX/Flax event](https://discuss.huggingface.co/t/open-to-the-community-community-week-using-jax-flax-for-nlp-cv/).
    """
)

st.sidebar.subheader("Configurable parameters")

max_len = st.sidebar.number_input(
    "Maximum length",
    value=8,
    help="The maximum length of the sequence (caption) to be generated."
)

num_beams = st.sidebar.number_input(
    "Number of beams",
    value=4,
    help="Number of beams for beam search. 1 means no beam search."
)

input_image = st.file_uploader("Insert image")
if st.button("Run"):
    with st.spinner(text="Getting results..."):
        if input_image:
            caption = process(image=input_image, server_url=backend, max_len=max_len, num_beams=num_beams)
            st.subheader("Result")
            st.write(caption.replace("<pad>", ""))
            st.text("English translation")
            st.write(translate(caption, "en", "id").replace("<pad>", ""))
        else:
            st.write("Please upload an image.")
