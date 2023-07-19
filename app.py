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
# Add Logo
#st.image("Ophy-Care.png", use_column_width=True)
# Add Logo in the Sidebar
st.sidebar.image("Ophy Care-01.png", use_column_width=True)

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
    This bot collects and analyzes user feedback to improve our services. We respect your privacy and ensure that your data is used solely for research purposes. We do not share your personal information with any third parties. If you have any concerns, please contact us at support@gmail.com.
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
if prompt := st.chat_input("Enter your feedback here"):
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
        response = "Thanks for your encouraging feedback! We are glad you liked our service."
        print(sentiment)
    elif sentiment[0] == 0:
        response = "We are sorry to hear that. We will work on improving our service."
        print(sentiment[0])
    else:
        response = "I'm not sure about the sentiment."

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Pink Background
st.markdown(
    """
    <style>
    .reportview-container {
        background-color: #87CEFA;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
