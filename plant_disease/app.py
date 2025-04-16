import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Loading the model
model = tf.keras.models.load_model('plant_disease_classifier.h5')

# Defining the classes of the leaves
class_names = ['Pepper Bell Bacterial Spot', 'Pepper Bell Healthy', 'Potato Early Blight', 'Potato Healthy', 'Tomato Bacterial Spot', 'Tomato Healthy']

st.title("Plant Disease Prediction")

uploaded_file = st.file_uploader("Upload a plant leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Resizing the image to 224 * 224 dimensions in order to match it with the input layer of the model
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"Predicted Plant Disease: **{predicted_class}**")

    st.write("Prediction Confidence:")
    for i, prob in enumerate(prediction[0]):
        st.write(f"{class_names[i]}: {prob:.2%}")
