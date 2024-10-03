import streamlit as st
import joblib
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from PIL import Image
import nltk

# Download stopwords if not available
nltk.download('stopwords')

# Load vectorizer and model
vectorizer = joblib.load("vectorizer.pkl")
model = joblib.load("fake_news_model.pkl")

# Initialize stemmer
port_stem = PorterStemmer()

# Function to perform text stemming
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    return ' '.join(stemmed_content)

# Streamlit app configuration
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# Custom styling
st.markdown(
    """
    <style>
        body {
            background-color: #f5f5f5;
            color: #333333;
        }
        .reportview-container {
            background: #ffffff;
            border-radius: 10px;
            padding: 2rem;
            box-shadow: 0px 4px 20px rgba(0, 0, 0, 0.1);
        }
        .stButton>button {
            color: white;
            background: linear-gradient(to right, #ff7e5f, #feb47b);
            border: none;
            border-radius: 4px;
            padding: 0.5rem 1rem;
            font-weight: bold;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and description
st.title("📰 Fake News Detector")
st.markdown("""
    Welcome to the Fake News Detector! 
    This application will help you determine whether a news story is **Fake** or **Real**. 
    Simply enter the content of the news below, and click **Check News**.
""")

# Add an image for aesthetics
image = Image.open("news_image.png")  # Make sure the image is present in the directory
st.image(image, use_column_width=True, caption="Can you trust the news?")

# Input for news content
st.write("### Please enter the news content:")
news_content = st.text_area("", placeholder="Type or paste the news article here...")

# Predict button
if st.button("Check News"):
    if news_content.strip():
        # Preprocess the input
        stemmed_content = stemming(news_content)

        # Transform input using the fitted vectorizer
        input_data = vectorizer.transform([stemmed_content])

        # Make a prediction
        prediction = model.predict(input_data)
        

        # Display result
        if prediction[0] == 0:
            st.success("🟢 The news is likely **Real**.")
            
        else:
            st.error("🔴 The news is likely **Fake**.")
    else:
        st.warning("⚠️ Please enter some news content.")

# Footer
st.markdown("""
    ---
    BY Krishna Tekwani!
""")
