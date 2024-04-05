import streamlit as st
import os
from PIL import Image
import tensorflow as tf
import numpy as np

# Function to load the saved model
@st.cache(allow_output_mutation=True)
def load_model():
    model_path = './model/modelo1.h5'
    model = tf.keras.models.load_model(model_path)
    return model

# Function to make predictions
def predict(image, model):
    # Preprocess the image
    image = np.array(image.resize((224, 224))) / 255.0  # Assuming input shape is (224, 224, 3)
    image = np.expand_dims(image, axis=0)

    # Make prediction
    prediction = model.predict(image)
    return prediction

def main():
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
