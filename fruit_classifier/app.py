import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Loading the model
model = tf.keras.models.load_model('fruit_classifier_scratch.h5')

# Defining the class names of the fruits
class_names = ['Apple fruit', 'Banana fruit', 'Cherry fruit', 'Chickoo fruit', 'Grapes fruit', 
               'Kiwi fruit', 'Mango fruit', 'Orange fruit', 'Strawberry fruit']

st.title("Fruit Classifier - CNN Model")

uploaded_file = st.file_uploader("Upload a fruit image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Converts the image to RGB Format
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Resizing to 128 * 128 as the input layer of the model takes 128 * 128 as input
    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"Predicted Fruit: **{predicted_class}**")

    st.write("Prediction Confidence:")
    for i, prob in enumerate(prediction[0]):
        st.write(f"{class_names[i]}: {prob:.2%}")
