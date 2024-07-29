import streamlit as st
from PIL import Image
import torch
import pickle
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import cv2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
PATH = "EfficientNet_B4NO2Model.pt"
my_model = torch.load(PATH, map_location='cpu')
my_model.eval()
# set logo and page title in browser
st.set_page_config(page_title='TARU',page_icon=':seedling:')
# Set the background color of the Streamlit using CSS
with open("style.css") as f:
    css = f.read()
st.markdown(f"""
    <style>
        {css}
    </style>
""", unsafe_allow_html=True)
# slide 1
with st.container():
    navbar_html = """
    <div class="navbar">
        <h2 style="float:left">[Taru.ai]</h2>
        <div>
            <a href="#home">Home</a>
            <a href="#about">About</a>
            <a href="#product">Product</a>
            <a href="#">Contact</a>
        </div>
    </div>
    """
    detail = """
    <div id="home" class="detail">
        <h1>DO YOU REALLY KNOW</h1>
        <h1 class='rice'>YOUR RICE <span>CROPS?</span></h1>
        <a href="#product" class="custom-button">Try out our latest Ai demo</a>
    </div>
    """
    st.markdown(navbar_html, unsafe_allow_html=True)
    st.markdown(detail, unsafe_allow_html=True)
    # You can call any Streamlit command, including custom components:
    st.image("picture.webp")
# Define a function to make predictions with the trained model
def predict(model, opencv_Image):
    # Load the image and transform it to the appropriate format
    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    #image = Image.open(image_path)
    pil_image = Image.fromarray(opencv_Image)
    image = transform(pil_image).unsqueeze(0).to(device)
    image = image

    # Make a prediction with the trained model
    #model.eval()
    with torch.no_grad():
        output = model(image)
        class_index = torch.argmax(output, dim=1).item()

    # Map the predicted index to the class name
    class_names = ['<div class="progress-container"><div class="color1"></div><div class="color2"></div><div class="color3"></div</div></div>Nitrogen deficiency Detected - <span>CLASS 1</span>  \n\n <span>Instructions -</span> \n Apply N-Fertilizer immediately', 
                   '<div class="progress-container"><div class="color2-class2"></div><div class="color3-class2"></div</div></div>Nitrogen deficiency Detected - <span>CLASS 2</span> \n\n <span>Instructions -</span> \n Apply N-Fertilizer soon', 
                   '<div class="progress-container"><div class="color3-class3"></div</div></div>Nitrogen deficiency Detected - <span>CLASS 3</span> \n\n <span>Instructions -</span> \n Do not apply N-Fertilizer and continue to monitor closely', 
                   '<div class="progress-container"><div class="color3-class3"></div</div></div>Nitrogen deficiency Detected - <span>CLASS 4</span> \n\n <span>Instructions -</span> \n Do not apply N-Fertilizer and continue to monitor']
    class_name = class_names[class_index]
    return class_name

# Set the background color of the Streamlit app to black using CSS
st.markdown("""
    <style>
    body {
        background-color: #000000;
    }
    </style>
""", unsafe_allow_html=True)



# Defining the rest of Streamlit app code
with st.container():
    # st.title("Nitrogen Deficiency for Rice Crop Prediction App")
    # st.write("Upload or take a photo of a rice leaf to see if it has nitrogen deficiency or not!")
    st.markdown('<div id="product"></div>', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["Upload Image", "Capture Image"])

    with tab1:
        test_image = st.file_uploader('Upload Image', type=['jpg', 'png','jpeg', 'jfif'] )
        st.write('Analyse')
        col1, col2 = st.columns(2)
        if test_image is not None:
            # Convert the file read to the bytes array.
            file_bytes = np.asarray(bytearray(test_image.read()), dtype=np.uint8)
            # Converting the byte array into opencv image. 0 for grayscale and 1 for bgr
            test_image_decoded = cv2.imdecode(file_bytes,1) 
            # col1.subheader('Uploaded Image')
            # col1.image(test_image_decoded, channels = "BGR")
            prediction = predict(my_model, test_image_decoded)
            # col2.subheader('Predicted Class')
            # col2.write(prediction)
            col2.markdown(prediction, unsafe_allow_html=True)
        st.write("lorem lorem lorem lorem lorem lorem lorem lorem lorem lorem lorem lorem lorem lorem lorem lorem lorem lorem")
    # with tab2:
    #     img_camera = st.camera_input("Capture Image")
        
    #     if img_camera is not None:
    #         # Convert the file read to the bytes array.
    #         file_bytes = np.asarray(bytearray(img_camera.read()), dtype=np.uint8)
    #         # Converting the byte array into opencv image. 0 for grayscale and 1 for bgr
    #         test_image_decoded = cv2.imdecode(file_bytes,1) 
    #         prediction = predict(my_model, test_image_decoded)
    #         st.subheader('Predicted Class')
    #         st.write(prediction)
