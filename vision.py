import time
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
import pathlib
import textwrap

import google.generativeai as genai

import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import tensorflow as tf
from keras.models import load_model
from keras import models
from keras.preprocessing import image
from keras.utils.image_utils import img_to_array, load_img 


os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


np.set_printoptions(suppress=True)
# Tải model
model = load_model("model/CNN.h5", compile=False)
class_names = open("model/labels.txt", "r").readlines()


## Function to load OpenAI model and get respones

def get_gemini_response(input,image):
    model = genai.GenerativeModel('gemini-pro-vision')
    if input!="":
       response = model.generate_content([input,image])
    else:
       response = model.generate_content(image)
    return response.text

##initialize our streamlit app

st.set_page_config(page_title="Identify Image")

st.header("Disease identification on fruit trees")


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

image=""

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", width=250)

if uploaded_file is not None:
    image = image.convert('RGB')
    image = image.resize((95, 95))
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    index = np.argmax(prediction)
    class_name = class_names[index]
    # Print prediction and confidence score
    st.write("Predict:")  
    st.title(class_name[2:])


input=st.text_area("Ask any question about the image in any language: ",key="input")

submit=st.button("Get answer")


if submit:
    with st.spinner('Loading...'):
        response=get_gemini_response(input,image)
        time.sleep(5)
    st.subheader("The Answer is")
    st.write(response)
