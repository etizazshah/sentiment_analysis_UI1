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

# Set the page configuration
st.set_page_config(
    page_title="Sentiment Analysis Bot",
    page_icon=":smiley:",  # Add your logo here or use a font-awesome icon name
    layout="centered",
    initial_sidebar_state="expanded",
)

# Load the vectorizer and model
@st.cache(allow_output_mutation=True)
def load_resources():
    vectorizer = pickle.load(open("./Models/count_vectorizer.pkl", "rb"))
    model = joblib.load(open("./Models/naive_model.pkl", "rb"))
    return vectorizer, model

text_vectorizer, naive_model = load_resources()

# Building the front end

st.title("Sentiment Analysis Bot")

# Page Description
st.write(
    """
    Welcome to our Sentiment Analysis Bot!

    This study aims to analyze the sentiment of your feedback or text and provide appropriate responses. 
    Feel free to interact with the bot and share your thoughts with us!

    For any privacy concerns, please read our privacy note below.
    """
)

# Privacy Note
st.info(
    """
    Privacy Note:
    This bot collects and analyzes user feedback to improve our services. We respect your privacy and ensure that your data is used solely for research purposes. We do not share your personal information with any third parties. If you have any concerns, please contact us at support@email.com.
    """
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.text_area("Enter your feedback here"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Analyze sentiment using your model
    new_text = preprocess_text([prompt])
    new_text = text_vectorizer.transform(new_text).toarray()
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

    # Clear text box after submission
    prompt = ""

# Pink Background
st.markdown(
    """
    <style>
    .reportview-container {
        background-color: #FFC0CB;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
