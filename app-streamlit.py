import streamlit as st
import os
from PIL import Image
from cnn import load_model_weights
from cnn import CNN
import torchvision
from pathlib import Path
import shutil


CLASSES = ['Bedroom', 'Coast', 'Forest', 'Highway', 'Industrial', 'Inside city', 'Kitchen', 'Living room', 'Mountain', 'Office', 'Open country', 'Store', 'Street', 'Suburb', 'Tall building']

def get_model_names(directory):
    model_names = []
    for filename in os.listdir(directory):
        if filename.endswith(".pt"):
            model_name = Path(filename).stem  # Get the file name without extension
            model_names.append(model_name)
    return model_names

MODELS = get_model_names('models')


# Function to load the saved model
@st.cache_data()
def load_model(model_name):

    model_weights = load_model_weights(model_name)
    my_trained_model = CNN(torchvision.models.resnet50(weights='DEFAULT'), 15) # 15 different classes
    my_trained_model.load_state_dict(model_weights)

    return my_trained_model


def predict(image, model):
    response = model.predict_single_image(image)
    return response

def show_feedback_select():
    st.session_state['show_feedback_form'] = True


def translate_output_class(output_class: int):
    return CLASSES[output_class]


def main():

    st.session_state['image_name'] = None
    st.session_state['save_path'] = None
    st.session_state['model_name'] = "resnet50-10epochs-2unfreezedlayers"
    st.session_state['show_feedback_form'] = False
    st.session_state['feedback_option'] = None
    st.session_state['show_feedback_form'] = False
    st.session_state['choice'] = None
    st.session_state['feedback_submitted'] = False

    favicon_path = "img\canonistia_logo.png" # Path to the favicon 
    st.set_page_config(page_title="Canonist.ia", page_icon=favicon_path, initial_sidebar_state="auto")
    style = "style='text-align: center;'"  # Define the style for the HTML elements
    # Define the HTML markup for the title with favicon
    title_with_favicon = f"""
        <head>
            <title>Canonist.ia: image classification of rooms using CNN</title>
            <link rel="shortcut icon" href="{favicon_path}">
        </head>
        <body>
            <h1 style='text-align: center;'>Canonist.ia: image classification of rooms using CNN</h1>
        </body>
    """

    # Render the title with favicon
    st.markdown(title_with_favicon, unsafe_allow_html=True)

    st.write(f"<p {style}>Your API for real estate portal image classification</p>", unsafe_allow_html=True)

    # Choose a model
    st.write(f"<h3 {style}>Choose a model</h3>", unsafe_allow_html=True)
    model_name = st.selectbox("Select a model", MODELS)

    # Subir imagen
    st.write(f"<h2 {style}>Upload your image</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        uploaded_file = st.file_uploader("Upload an image...", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.image(image, caption='Uploaded Image', use_column_width=True)

        # Save the image if it's jpg, jpeg, or png
        if uploaded_file.type.startswith('image'):
            st.session_state['image_name']  = uploaded_file.name
            st.session_state['save_path'] = os.path.join('tmp', st.session_state['image_name'] )
            image.save(st.session_state['save_path'])
            st.success('Image saved successfully!')

            if model_name is not None:
                # Load the model
                model = load_model(model_name)
                # Make predictions
                prediction = predict(image, model)
                confidence = 75.42
                # Display the prediction result
                st.write(f"<h3 {style}>The model {model_name} is {confidence}% sure that your image is a...<br>{CLASSES[prediction]}</h3>", unsafe_allow_html=True)

                # Mostrar el widget radio solo si no se ha seleccionado una opción
                if st.session_state['feedback_option'] is None:
                    feedback_option = st.radio("Feedback:", ['✔️', '❌'], index=None)

                    # Actualizar el estado de sesión basado en la selección
                    if feedback_option in ['✔️', '❌']:
                        st.session_state['feedback_option'] = feedback_option

                        # Mensaje de agradecimiento por el feedback positivo
                        if feedback_option == '✔️':
                            st.session_state['choice'] = CLASSES[prediction]

                # Manejar la lógica de mostrar el formulario de feedback si se eligió la opción negativa
                if st.session_state['feedback_option'] == '❌':
                    st.session_state['show_feedback_form'] = True
                else:
                    st.session_state['show_feedback_form'] = False

                if st.session_state.get('show_feedback_form'):
                    st.write("What was actually your image?")
                    choice = st.selectbox("Select the correct option:", CLASSES)
                    st.session_state['choice'] = choice
                    
                if st.button('Submit Feedback'):
                    new_folder = os.path.join('dataset', 'new', st.session_state['choice'])
                    if not os.path.exists(new_folder):
                        os.makedirs(new_folder)
                    new_path = os.path.join(new_folder, st.session_state['image_name'])
                    shutil.copyfile(st.session_state['save_path'], new_path)
                    st.session_state['feedback_submitted'] = True
                    st.success('Feedback submitted successfully!')




if __name__ == "__main__":
    main()