import streamlit as st
import os
from PIL import Image
import torch
import numpy as np
import torch.nn.functional as F
from cnn import load_model_weights
from cnn import CNN
import torchvision

# Function to load the saved model
@st.cache_data()
def load_model():
    # Load model
    # models\resnet50-10epochs-2unfreezedlayers.pt
    model_weights = load_model_weights('resnet50-10epochs-2unfreezedlayers')
    my_trained_model = CNN(torchvision.models.resnet50(weights='DEFAULT'), 15) # 15 different classes
    my_trained_model.load_state_dict(model_weights)

    return my_trained_model


def predict(image, model):
    response = model.predict_single_image(image)

    return response



def main():
    favicon_path = "img\canonistia_logo.png" # Path to the favicon 
    st.set_page_config(page_title="Canonist.ia", page_icon=favicon_path)
    st.title('Image Classification of rooms using CNN')

    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Save the image if it's jpg, jpeg, or png
        if uploaded_file.type.startswith('image'):
            save_path = os.path.join('./tmp', uploaded_file.name)
            image.save(save_path)
            st.success('Image saved successfully!')

            # Load the model
            model = load_model()

            # Make predictions
            prediction = predict(image, model)

            # Display the prediction result
            st.write('Prediction:', prediction)


if __name__ == "__main__":
    main()