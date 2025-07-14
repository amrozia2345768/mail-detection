import streamlit as st
import joblib
import base64
import numpy as np
import re

# Load Model & Vectorizer
model = joblib.load('spam_pipeline.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Rule-based Spam Check
spam_keywords = ["earn", "work from home", "click here", "congratulations", "prize", "urgent", "free", "money"]

def spam_flag(text):
    text = text.lower()
    return int(any(word in text for word in spam_keywords))

# Add Background Image
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url(data:image/jpg;base64,{encoded});
             background-size: cover;
             background-repeat: no-repeat;
             background-position: center;
             color: white;
         }}
         .stTextArea label {{
             color: white !important;
             font-size: 18px !important;
             font-weight: 600 !important;
         }}
         .stTextArea textarea {{
             background-color: #ffffff !important;
             color: #000000 !important;
             font-size: 16px !important;
             border-radius: 8px;
         }}
         .stButton > button {{
             background-color: #ff4b4b;
             color: white;
             font-weight: bold;
             font-size: 16px;
             border-radius: 8px;
         }}
         .stMarkdown h1 {{
             color: white;
             text-align: center;
             text-shadow: 1px 1px 4px black;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_local('background.jpg')

# Title
st.markdown("<h1>📧 Hybrid Email Spam Detector</h1>", unsafe_allow_html=True)

# Input Area
email = st.text_area("✉️ Enter Email Text Below", height=150)

if st.button("🚀 Predict"):
    if email.strip() == "":
        st.warning("⚠️ Please enter some text!")
    else:
        # Preprocess input
        clean = re.sub(r'\W', ' ', email.lower())
        tfidf_input = vectorizer.transform([clean])
        rule_input = np.array([spam_flag(clean)]).reshape(-1,1)

        from scipy.sparse import hstack
        combined_input = hstack((tfidf_input, rule_input))

        # Predict
        result = model.predict(combined_input)[0]
        proba = model.predict_proba(combined_input)[0][1]

        # Display output with dynamic color
        if result == 1:
            st.markdown(f"<h3 style='color:#FF0000;'>🚫 SPAM Email Detected!</h3>", unsafe_allow_html=True)
            st.markdown(f"<b style='color:#FF0000;'>Spam Probability: {round(proba*100,2)}%</b>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h3 style='color:#006400;'>✅ Safe Email (HAM)</h3>", unsafe_allow_html=True)
            st.markdown(f"<b style='color:#006400;'>Spam Probability: {round(proba*100,2)}%</b>", unsafe_allow_html=True)

        st.progress(min(int(proba*100), 100))

# Footer Light Grey & Slight Bold
st.markdown("---")
st.markdown(
    """
    <p style='text-align:center; color:#D3D3D3; font-size:14px; font-weight:500;'>
    Made with ❤️ by <b>Amrozia</b> | Data Zenix Solution
    </p>
    """,
    unsafe_allow_html=True
)
