# !pip install scikit-learn nltk matplotlib pandas gensim joblib --quiet
import random
import time
import joblib
import pickle
import streamlit as st
from preprocessor import preprocess_text
import nltk
nltk.download('punkt')
nltk.download('wordnet')

# Function to clear text box after submission
def clear_text():
    st.session_state.feedback_text = ""

# Function to handle user feedback
def handle_feedback():
    if st.session_state.feedback_text:
        # Display user message in chat message container
        st.chat_message("user").markdown(st.session_state.feedback_text)
        # Analyze sentiment using your model
        new_text = preprocess_text(st.session_state.feedback_text)
        new_text = text_vectorizer.transform([new_text]).toarray()
        sentiment = naive_model.predict(new_text)  # Replace `predict()` with the appropriate method for sentiment analysis

        # Determine sentiment label
        if sentiment[0] == 1:
            response = "Thanks for your encouraging feedback! We are glad you liked our service."
        elif sentiment[0] == 0:
            response = "We are sorry to hear that. We will work on improving our service."
        else:
            response = "I'm not sure about the sentiment. Could you provide more details?"

        # Display assistant response in chat message container
        st.chat_message("assistant").markdown(response)

        # Clear text box
        clear_text()

# Load the saved model from a file
@st.cache_resource
def vec():
    with open("./Models/count_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return vectorizer

text_vectorizer = vec()

@st.cache_resource
def get_model():
    model = joblib.load("./Models/naive_model.pkl")
    return model

naive_model = get_model()

# Set pink background
st.markdown(
    """
    <style>
    body {
        background-color: #FFC0CB;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add a help page
st.sidebar.title("Help")
st.sidebar.info("Welcome to our Sentiment Analysis Bot for Research! Type in your feedback in the textbox below and click the Submit button. The bot will analyze the sentiment of your feedback and respond accordingly. Enjoy your interaction!")

# Add random logo
st.sidebar.image("https://example.com/random_logo.png", use_column_width=True)

# Building the front end

st.title("Sentiment Analysis Bot for Research")
st.markdown("Welcome to our Sentiment Analysis Bot! Type in your feedback below to see how our bot responds.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Text box for user input
if "feedback_text" not in st.session_state:
    st.session_state.feedback_text = ""

feedback_text = st.text_area("Your Feedback", height=100, max_chars=None, value=st.session_state.feedback_text)

# Submit button
if st.button("Submit"):
    handle_feedback()

# React to user input
if prompt := feedback_text:
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Analyze sentiment using your model
    new_text = preprocess_text(prompt)
    new_text = text_vectorizer.transform([new_text]).toarray()
    sentiment = naive_model.predict(new_text)  # Replace `predict()` with the appropriate method for sentiment analysis

    # Determine sentiment label
    if sentiment[0] == 1:
        response = "Thanks for your encouraging feedback! We are glad you liked our service."
    elif sentiment[0] == 0:
        response = "We are sorry to hear that. We will work on improving our service."
    else:
        response = "I'm not sure about the sentiment. Could you provide more details?"

    # Display assistant response in chat message container
    st.chat_message("assistant").markdown(response)

    # Clear text box
    clear_text()
