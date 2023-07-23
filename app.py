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

# Define a set of additional stopwords
additional_stopwords = {"how", "why", "other", "similar", "words"}

# Set the threshold value for neutral sentiment
neutral_threshold = 0.1  # Adjust this threshold as needed

# React to user input
if prompt := st.chat_input("Enter your feedback here"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)

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
            # Ask for more details for proper sentiment analysis
            response = "Your feedback is appreciated! However, it seems like you've provided only one word. Please share more details for better analysis."
        else:
            # Check if the input contains words like "why" or "how"
            if any(word.lower() in additional_stopwords for word in new_text_preprocessed.split()):
                # Ask for more feedback to understand the context
                response = "Thank you for your feedback! To better understand your context, could you provide more details or explain 'why' or 'how' you feel this way?"
                # Store the additional feedback in the session
                st.session_state.additional_feedback = prompt
            else:
                # Perform sentiment analysis on the main feedback
                new_text_vectorized = text_vectorizer.transform([new_text_preprocessed]).toarray()
                sentiment = naive_model.predict_proba(new_text_vectorized)  # Use predict_proba to get the probability of each class

                # Determine sentiment label
                if "like" in new_text_preprocessed.lower():
                    # Set sentiment to positive if the word "like" is present
                    sentiment_label = 1
                else:
                    # Check if sentiment is neutral based on threshold
                    if abs(sentiment[0][1] - 0.5) <= neutral_threshold:
                        sentiment_label = -1  # Neutral sentiment
                    else:
                        sentiment_label = sentiment[0][1] > 0.5  # Positive sentiment if probability > 0.5, otherwise negative

                if sentiment_label == 1:
                    response = "Thanks for showing positivity!"
                elif sentiment_label == 0:
                    response = "We are sorry to hear that."
                else:
                    response = "I did not get your point. Can you please elaborate?"

        # Display assistant response in chat message container
        st.chat_message("assistant").markdown(response)

    except Exception as e:
        # Handle any errors that occur during preprocessing or prediction
        error_message = f"Error: {str(e)}"
        st.chat_message("assistant").markdown(error_message)

# Add both user and assistant responses to the chat history
st.session_state.messages.append({"role": "user", "content": prompt})
st.session_state.messages.append({"role": "assistant", "content": response})

# If additional feedback is present, analyze it as a new entry
if hasattr(st.session_state, "additional_feedback"):
    with st.chat_message("assistant"):
        st.markdown("Analyzing additional feedback:")
        st.chat_message("user").markdown(st.session_state.additional_feedback)

    try:
        # Analyze additional feedback using your model
        new_text_preprocessed = preprocess_text(st.session
