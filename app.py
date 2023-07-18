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


@st.cache_resource
def vec():
    # Load the saved model from a file
    with open("./Models/count_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return vectorizer

text_vectorizer = vec()

@st.cache_resource
def get_model():
    model = joblib.load("./Models/naive_model.pkl")
    return model

naive_model = get_model()

# Building the front end

st.title("Sentiment Bot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Enter your text here"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Analyze sentiment using your model
    new_text = preprocess_text(prompt)
    new_text = text_vectorizer.transform(new_text).toarray()
    sentiment = naive_model.predict(new_text)  # Replace `predict()` with the appropriate method for sentiment analysis

    # Determine sentiment label
    if sentiment[0] == 1:
        response = "I sense a positive sentiment!"
        print(sentiment)
    elif sentiment[0] == 0:
        response = "I sense a negative sentiment!"
        print(sentiment[0])
    else:
        response = "I'm not sure about the sentiment."

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
