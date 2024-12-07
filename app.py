import keras
from keras.models import load_model
import streamlit as st
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image

# Custom CSS for a modern look
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(to right, #ece9e6, #ffffff);
        font-family: 'Roboto', sans-serif;
        color: #333;
    }
    .header {
        text-align: center;
        margin: 20px 0;
        padding: 20px;
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .header h1 {
        font-size: 2.5em;
        color: #4CAF50;
    }
    .header p {
        font-size: 1.2em;
        color: #555;
    }
    .file-upload {
        text-align: center;
        margin: 20px 0;
        padding: 20px;
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .result-card {
        margin-top: 30px;
        text-align: center;
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 6px 10px rgba(0, 0, 0, 0.15);
        animation: zoomIn 0.5s ease-out;
    }
    .result-card h2 {
        font-size: 2em;
        color: #FF5722;
    }
    .result-card p {
        font-size: 1.2em;
        color: #333;
    }
    .btn {
        display: inline-block;
        margin: 10px;
        padding: 10px 20px;
        background-color: #4CAF50;
        color: white;
        text-decoration: none;
        border-radius: 5px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .btn:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }
    @keyframes zoomIn {
        from {
            transform: scale(0.8);
            opacity: 0;
        }
        to {
            transform: scale(1);
            opacity: 1;
        }
    }
    h3{
        color:#008000;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# App Header
st.markdown(
    """
    <div class="header">
        <h1>🌼 Flower Classification 🌻</h1>
        <p>Upload a flower image and let AI identify it for you!</p>
    </div>
    """,
    unsafe_allow_html=True,
)

flower_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# Load the pre-trained model
model = load_model('Flower_Recog_Model.h5')

def classify_images(image_bytes):
    input_image = Image.open(BytesIO(image_bytes)).resize((180, 180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)

    # Predict the class
    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    class_name = flower_names[np.argmax(result)]
    confidence = np.max(result) * 100
    return class_name, confidence

# File Uploader Section
st.markdown(
    """
    <div class="file-upload">
        <h3 >Upload Your Flower Image</h3>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader("Choose an image file")

if uploaded_file is not None:
    # Read the uploaded file as bytes
    image_bytes = uploaded_file.read()

    # Display the uploaded image
    st.image(image_bytes, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing image..."):
        # Perform classification
        class_name, confidence = classify_images(image_bytes)

    # Display the classification result with animations
    st.markdown(
        f"""
        <div class="result-card">
            <h2>🌸 Identified Flower: <b>{class_name}</b> 🌸</h2>
            <p>Confidence Level: <b>{confidence:.2f}%</b></p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Footer
st.markdown(
    """
    <hr>
    <footer style="text-align: center; font-size: 14px;">
        🌺 Powered by AI | Made with ❤️ by Sayandip Bera
    </footer>
    """,
    unsafe_allow_html=True,
)
