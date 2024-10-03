import streamlit as st
import joblib
import pandas as pd
from PIL import Image


model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")


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
        .css-1d391kg {
            background-color: #ffffff;
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


st.title("📰 Fake News Detector")
st.markdown("""
    Welcome to the Fake News Detector! 
    This application will help you determine whether a news story is **Fake** or **Real**. 
    Simply enter the content of the news below, and click **Check News**.
""")


image = Image.open("news_image.jpg") 
st.image(image, use_column_width=True, caption="Can you trust the news?")


st.write("### Please enter the news content:")
news_content = st.text_area("", placeholder="Type or paste the news article here...")


if st.button("Check News"):
    if news_content.strip():
        
        input_data = vectorizer.transform([news_content])
        
        
        prediction = model.predict(input_data)
        
        
        if prediction[0] == 1:
            st.success("🟢 The news is likely **Real**.")
        else:
            st.error("🔴 The news is likely **Fake**.")
    else:
        st.warning("⚠️ Please enter some news content.")


st.markdown("""
    ---
    BY Krishna Tekwani!
""")

