import streamlit as st
import os
from PIL import Image
from cnn import load_model_weights
from cnn import CNN
import torchvision


CLASSES = ['Bedroom', 'Coast', 'Forest', 'Highway', 'Industrial', 'Inside city', 'Kitchen', 'Living room', 'Mountain', 'Office', 'Open country', 'Store', 'Street', 'Suburb', 'Tall building']
MODELS = ['resnet50-1epoch-one-layer-unfreezed',
          'resnet50-1epoch',
          'resnet50-10epochs-2unfreezedlayers']


# Function to load the saved model
@st.cache_data()
def load_model(model_name):
    # Load model
    model_weights = load_model_weights(model_name)
    my_trained_model = CNN(torchvision.models.resnet50(weights='DEFAULT'), 15)
    # models\resnet50-10epochs-2unfreezedlayers.pt
    model_weights = load_model_weights('resnet50-10epochs-2unfreezedlayers')
    my_trained_model = CNN(torchvision.models.resnet50(weights='DEFAULT'), 15) # 15 different classes
    my_trained_model.load_state_dict(model_weights)
    return my_trained_model


def predict(image, model):
    response = model.predict_single_image(image)
    return response

def show_feedback_select():
    st.session_state['show_feedback_form'] = True

def prueba():
    print("AAAAAAAAAAAAAAAAAAAAAAAAAAA")
    st.session_state['feedback_choice'] = "a"


def main():

    st.session_state['image_name'] = None
    st.session_state['save_path'] = None
    st.session_state['model_name'] = None
    st.session_state['show_feedback_form'] = False
    st.session_state['feedback_choice'] = None
    print(st.session_state)
    # Configuración de la página para usar el ancho completo y centrar
    favicon_path = "img/canonistia_logo.png" # Path to the favicon 
    st.set_page_config(layout="wide", page_title="Canonist.ia", page_icon=favicon_path)
    style = "style='text-align: center;'"

def translate_output_class(output_class: int):
    classes = ['bedroom', 'coast', 'forest', 'highway', 'industrial', 'inside city', 'kitchen', 'living room', 
               'mountain', 'office', 'open country', 'store', 'street', 'suburb', 'tall building']
    return classes[output_class]

def main():
    favicon_path = "img\canonistia_logo.png" # Path to the favicon 
    st.set_page_config(page_title="Canonist.ia", page_icon=favicon_path, initial_sidebar_state="expanded")
    style = "style='text-align: center;'"
    # Logo of the application and title, next to each other
    logo_path = "img/canonistia_logo.png"
    col1, col2 = st.columns([1, 6])
    with col1:
        st.image(logo_path, width=100)
    with col2:
        st.write(f"<h1>Canonist.ia</h1>", unsafe_allow_html=True)
    
    st.write(f"<p>Your API for real estate portal image classification</p>", unsafe_allow_html=True)

    # Subir imagen
    st.write(f"<h2>Upload your image</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=['png', 'jpg', 'jpeg'])
    
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
            # Choose a model
            st.write(f"<h3 {style}>Choose a model</h3>", unsafe_allow_html=True)
            model_name = st.selectbox("Select a model", MODELS, index=None, placeholder='Choose a model')
            if model_name is not None:
                # Load the model
                model = load_model(model_name)
                # Make predictions
                prediction = predict(image, model)
                # confidence = F.softmax(prediction, dim=1).max().item() * 100
                # Display the prediction result
                st.write(f"<h3 {style}>Your image is a...<br>{CLASSES[prediction]}</h3>", unsafe_allow_html=True)
                if st.button('✔️'):
                    st.success('Thank you for your feedback!')
                # st.button('❌', on_click=show_feedback_select)
                if st.button('❌'):
                    st.session_state['show_feedback_form'] = True
                if st.session_state.get('show_feedback_form'):
                    st.write("What was actually your image?")
                    choice = st.selectbox("Select the correct option:", CLASSES, index=None, placeholder='Choose an option', on_change=prueba)
                    print("AAAAAAAAAAAAAAAAAAAAAAAAAAA")
                    print(choice)
                    if st.session_state['feedback_choice'] is not None:
                        print("AAAAAAAAAAAAAAAAAAAAAAAAAAA")
                        new_folder = os.path.join('dataset', 'new', choice)
                        if not os.path.exists(new_folder):
                            os.makedirs(new_folder)
                        new_path = os.path.join(new_folder, st.session_state['image_name'])
                        # os.rename(st.session_state['save_path'], new_path)
                        os.copyfile(st.session_state['save_path'], new_path)
                        st.success('Thank you for your feedback!')



if __name__ == "__main__":
    main()