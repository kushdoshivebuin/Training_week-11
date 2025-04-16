import streamlit as st
import numpy as np
from tensorflow import keras
from keras.datasets import imdb

# Load the IMDB word index
word_index = imdb.get_word_index()
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

# Constants
MAX_LEN = 500
VOCAB_SIZE = 10000

# Load the trained model
model = keras.models.load_model("imdb_sentiment.h5")

# Function to encode the input review
def encode_review(review, max_len=MAX_LEN):
    words = review.lower().split()
    encoded = [word_index.get(word, 2) for word in words if word_index.get(word, 2) < VOCAB_SIZE]
    if len(encoded) > max_len:
        encoded = encoded[:max_len]
    else:
        encoded = [0] * (max_len - len(encoded)) + encoded  
    return np.array([encoded])  

st.set_page_config(page_title="IMDB Sentiment Analyzer 🎬", page_icon="🎭")
st.title("🎬 IMDB Movie Review Sentiment Analyzer")
st.markdown("Enter a movie review below, and the model will predict if it's **positive** or **negative**.")

# Input box
user_input = st.text_area("📝 Write your review here:", height=200)

if st.button("Analyze Sentiment"):
    if not user_input.strip():
        st.warning("Please enter a review to analyze.")
    else:
        input_array = encode_review(user_input)
        prediction = model.predict(input_array)[0][0]

        if prediction >= 0.5:
            st.success(f"Positive Sentiment (Confidence: {prediction:.2f})")
        else:
            st.error(f"Negative Sentiment (Confidence: {1 - prediction:.2f})")