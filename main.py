import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image

# Load Model
MODEL_PATH = os.path.join(os.getcwd(), "trained_mobilenet_model.keras")  # Updated model path

if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    model = None
    st.error("Trained model file not found! Ensure 'trained_mobilenet_model.keras' exists in the project directory.")

# Model Prediction Function
def model_prediction(test_image):
    if test_image is not None:
        image = Image.open(test_image)
        image = image.resize((128, 128))  # Resize for model input
        input_arr = np.array(image) / 255.0  # Normalize pixel values
        input_arr = np.expand_dims(input_arr, axis=0)  # Add batch dimension

        prediction = model.predict(input_arr)
        result_index = np.argmax(prediction)
        return result_index
    else:
        return None

# Sidebar Navigation
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Home Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = os.path.join(os.getcwd(), "home_page.jpeg")

    if os.path.exists(image_path):
        st.image(image_path, use_container_width=True)
    else:
        st.error("Home page image not found! Please check the file path.")

    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍

    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

# About Page
elif app_mode == "About":
    st.header("About")
    st.markdown("""
    #### About Dataset
    This dataset consists of about 87K RGB images of healthy and diseased crop leaves, categorized into 38 different classes.

    #### Content
    - **Train Set:** 70,295 images
    - **Validation Set:** 17,572 images
    - **Test Set:** 33 images
    
    #### Creators
    Aman Jain, Ayush Prakash, Rupam Das, Udeshna Saikia, Sahil Suman
    """)

# Disease Recognition Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])

    if test_image is not None:
        st.image(test_image, use_container_width=True)

    if st.button("Predict"):
        if model is None:
            st.error("Model not loaded! Please check if 'trained_mobilenet_model.keras' is in the directory.")
        else:
            with st.spinner("Please Wait.."):
                result_index = model_prediction(test_image)
                if result_index is not None:
                    class_names = [
                        'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
                        'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                        'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                        'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
                        'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
                        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
                        'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                        'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                        'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                        'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                        'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                        'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                        'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                        'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                        'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
                    ]
                    st.success(f"Model is Predicting: **{class_names[result_index]}**")
                else:
                    st.error("Error in processing the image. Please upload a valid file.")
