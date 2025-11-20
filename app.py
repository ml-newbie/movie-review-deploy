import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU only

import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.run_functions_eagerly(True)

import streamlit as st
import numpy as np
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb


# ---------------------------
# 1. CONSTANTS
# ---------------------------
MAX_WORDS = 10000     # Must match training
MAX_LEN = 100         # Must match training


# ---------------------------
# 2. Load IMDb word index
# ---------------------------
word_index = imdb.get_word_index()
word_index = {k.lower(): (v + 3) for k, v in word_index.items()}  # shift because 0,1,2 reserved
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2


# ---------------------------
# 3. Cache model (CRITICAL)
# ---------------------------
@st.cache_resource
def load_sentiment_model():
    return load_model("saved_model/sentiment_model.keras")

model = load_sentiment_model()


# ---------------------------
# 4. Preprocessing function
# ---------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s']", "", text)
    words = text.split()

    seq = []
    for w in words:
        idx = word_index.get(w, 2)  # unknown = 2
        if idx >= MAX_WORDS:        # Too rare → map to <UNK>
            idx = 2
        seq.append(idx)

    seq = np.array(seq, dtype="int32")
    seq = pad_sequences([seq], maxlen=MAX_LEN)
    return seq


# ---------------------------
# 5. Streamlit UI
# ---------------------------
st.set_page_config(page_title="Movie Sentiment Analyzer")

st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Enter a movie review below and the model will predict whether it's **positive** or **negative**.")


# ---------------------------
# 6. Input Box
# ---------------------------
review = st.text_area("Write your review here:", height=200)

if st.button("Predict Sentiment"):
    if not review.strip():
        st.warning("Please enter a review before predicting.")
    else:
        seq = preprocess_text(review)
        pred = model.predict(seq)[0][0]

        if pred >= 0.5:
            sentiment = "Positive 😀"
            color = "#d1f5d3"  # light green
            confidence = pred
        else:
            sentiment = "Negative 😞"
            color = "#f5d1d1"  # light red
            confidence = 1 - pred

        # Styled output box
        st.markdown(
            f"""
            <div style='background-color:{color};
                        padding:20px;
                        border-radius:10px;
                        margin-top:20px;
                        text-align:center;
                        font-size:20px;'>
                <b>{sentiment}</b><br>
                Confidence: {confidence:.2f}
            </div>
            """,
            unsafe_allow_html=True
        )
'''
    # -------------------------------- #
    # 7. GitHub Deployment Instructions
    # -------------------------------- #
    First, create a new repository on GitHub named `movie-review-deploy`.
    git init
    git add .
    git commit -m "Initial commit"
    git branch -M main
    git remote add origin https://github.com/yourusername/movie-review-deploy.git
    git push -u origin main
'''