import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
from PIL import Image

# Load model
model = load_model("face_mask_cnn_model.h5")

# Title
st.title("😷 Face Mask Detection Web App")

# File uploader
uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

# Prediction function
def predict_mask(image):
    image = image.convert('RGB')
    image_np = np.array(image)
    image_resized = cv2.resize(image_np, (96, 96))
    image_array = img_to_array(image_resized) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array)[0][0]
    label = "Mask 😷" if prediction < 0.5 else "No Mask 😐"
    confidence = (1 - prediction) if prediction < 0.5 else prediction
    confidence = round(confidence * 100, 2)

    return label, confidence, image_np

# Display image and result
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    label, confidence, display_image = predict_mask(image)

    st.image(display_image, caption=f"Prediction: {label} ({confidence}%)", use_column_width=True)
    st.success(f"✅ Prediction: {label} with {confidence}% confidence")
