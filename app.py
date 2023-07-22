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

st.sidebar.image("Ophy-Care.jpg", use_column_width=True)
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-color: #DFF0D8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Company Information
st.sidebar.write(
    """
    **OphyCare's Mission:**
    OphyCare is a post-revenue early stage company building Cloud Based Electronic Medical Record (EMR) for International Medical Relief Organizations like "Doctors without Borders". 90% of medical mission and relief organizations don't have proper software to manage patient data. More than 50% of these organizations use pen and paper and the ones who digitally store the data, use MS word or Google Drive. We are providing EPIC like EMR to these organizations where they can store the patient data which is accessible to doctors anywhere from the world. Along with storing patient data, it also gives live updates and statistical data to donors to track their money to the exact patient it was spent on. We have a huge traction and have launched our product with Palestine Children's Relief Fund with great success. We are looking for strategic partners, investors and advisors to join us.
    """
)

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


# Initialize a variable to store additional feedback
additional_feedback = ""

# React to user input
if prompt := st.chat_input("Enter your feedback here"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        # Check if the prompt is a string, if not, display an error message
        if not isinstance(prompt, str):
            raise ValueError("Invalid input! Please enter a valid text.")

        # Analyze sentiment using your model
        new_text_preprocessed = preprocess_text(prompt)

        # Check if the preprocessing resulted in an empty string
        if not new_text_preprocessed:
            raise ValueError("Invalid input! Please enter a valid text.")

        # Check if the input contains only one word
        if len(new_text_preprocessed.split()) == 1:
            # Handle cases where certain words should not show sentiment
            if new_text_preprocessed.lower() in ["how", "similar_words_here"]:
                additional_feedback = st.text_input("Please provide more details for proper sentiment analysis:")
                st.session_state.messages.append({"role": "assistant", "content": "Sure, please provide more details for proper sentiment analysis."})
            else:
                new_text_vectorized = text_vectorizer.transform([new_text_preprocessed]).toarray()
                sentiment = naive_model.predict(new_text_vectorized)  # Replace `predict()` with the appropriate method for sentiment analysis

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

        else:
            # Proceed with the usual sentiment analysis
            new_text_vectorized = text_vectorizer.transform([new_text_preprocessed]).toarray()
            sentiment = naive_model.predict(new_text_vectorized)  # Replace `predict()` with the appropriate method for sentiment analysis

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

    except Exception as e:
        # Handle any errors that occur during preprocessing or prediction
        error_message = f"Error: {str(e)}"
        with st.chat_message("assistant"):
            st.markdown(error_message)
        # Add error message to chat history
        st.session_state.messages.append({"role": "assistant", "content": error_message})

# Assess the sentiment of the additional feedback provided by the user
if additional_feedback:
    additional_text_preprocessed = preprocess_text(additional_feedback)
    additional_text_vectorized = text_vectorizer.transform([additional_text_preprocessed]).toarray()
    sentiment = naive_model.predict(additional_text_vectorized)  # Replace `predict()` with the appropriate method for sentiment analysis

    # Determine sentiment label
    if sentiment[0] == 1:
        additional_response = "Thanks for your encouraging feedback! We are glad you liked our service."
        print(sentiment)
    elif sentiment[0] == 0:
        additional_response = "We are sorry to hear that. We will work on improving our service."
        print(sentiment[0])
    else:
        additional_response = "I'm not sure about the sentiment."

    # Display additional assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(additional_response)
    # Add additional assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": additional_response})

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
