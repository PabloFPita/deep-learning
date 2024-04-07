import streamlit as st
import os
from streamlit_modal import Modal
import streamlit as st
import os
from PIL import Image
import torch
import numpy as np
import torch.nn.functional as F
from cnn import load_model_weights
from cnn import CNN
import torchvision

# Configuración de la página para usar el ancho completo
st.set_page_config(layout="wide")

style = "style='text-align: center;'"

# Título de tu aplicación
st.write(f"<h1 {style}>Canonist.ia</h1>", unsafe_allow_html=True)
st.write(f"<p {style}>Your API for real estate portal image classification</p>", unsafe_allow_html=True)

# Subir imagen
st.write(f"<h2 {style}>Upload your image</h2>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1,2,1])
with col2:
    uploaded_image = st.file_uploader("", type=['png', 'jpg', 'jpeg'])

result = "Backyard"
confidence = 75
# Function to load the saved model
@st.cache_data()
def load_model():
    # Load model
    # models\resnet50-1epoch-one-layer-unfreezed.pt
    model_weights = load_model_weights('resnet50-1epoch-one-layer-unfreezed')
    my_trained_model = CNN(torchvision.models.resnet50(weights='DEFAULT'), 15) # 15 different classes
    my_trained_model.load_state_dict(model_weights)

    return my_trained_model


def predict(image, model):
    response = model.predict_single_image(image)

    return response
# Mostrar el porcentaje de confianza y la clase predicha
st.write(f"<h3 {style}>We are {confidence}% sure that your image is a...<br>{result}</h3>", unsafe_allow_html=True)



cols = st.columns(10)

with cols[4]:
    if st.button('✔️'):
        st.toast('Thank you for your feedback!', icon='😍')

modal = Modal(
    "Give us feedback!", 
    key="demo-modal",
    # Optional
    padding=15,    # default value
    max_width=744  # default value
)

if modal.is_open():
    with modal.container():
        st.write("What was actually your image?")
        options = ['Living Room', 'Kitchen', 'Bathroom', 'Bedroom', 'Front Yard', 'Other']
        choice = st.radio("Select the correct option:", options)
        if st.button('Submit'):
            st.write(f"You indicated the image is: {choice}. Thank you for helping us improve!")

with cols[5]:
    if st.button('❌'):
        modal.open()   