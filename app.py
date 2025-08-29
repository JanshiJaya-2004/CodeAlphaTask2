# app.py
import os, string
import numpy as np
import pandas as pd
import streamlit as st

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources if missing
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab')
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# --- Preprocessing ---
STOPWORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text: str) -> str:
    """Lowercase, remove punctuation, tokenize, remove stopwords, lemmatize"""
    if not isinstance(text, str):
        return ""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha() and t not in STOPWORDS]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

# --- Streamlit UI ---
st.set_page_config(page_title="FAQ Chatbot", layout="wide")
st.title("💬 FAQ Chatbot")

# Upload or use sample data
uploaded = st.file_uploader("Upload a FAQ CSV (with 'question','answer')", type=["csv"])

if uploaded:
    df = pd.faqs_csv(uploaded)
elif os.path.exists("faqs.csv"):
    st.info("Using faqs.csv from project folder")
    df = pd.read_csv("faqs.csv")
else:
    st.info("Using sample FAQs")
    df = pd.DataFrame({
        "question": [
            "What is your refund policy?",
            "How do I reset my password?",
            "Do you ship internationally?",
            "How do I contact support?",
            "What payment methods do you accept?"
        ],
        "answer": [
            "We offer a 30-day refund. Contact support@company.com for details.",
            "Click 'Forgot password' on the login page and follow the instructions.",
            "Yes, we ship worldwide. Shipping charges vary by location.",
            "You can contact support at support@company.com.",
            "We accept Visa, MasterCard, PayPal, and American Express."
        ]
    })

# Validate CSV
if 'question' not in df.columns or 'answer' not in df.columns:
    st.error("CSV must contain 'question' and 'answer' columns")
    st.stop()

# Preprocess questions
df['question_clean'] = df['question'].apply(preprocess)
# Build TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['question_clean'])

# User input
st.subheader("Ask me something:")
user_q = st.text_area("Your question", height=100)

if st.button("Get Answer"):
    if not user_q.strip():
        st.warning("Please enter a question.")
    else:
        q_clean = preprocess(user_q)
        q_vec = vectorizer.transform([q_clean])
        sims = cosine_similarity(q_vec, tfidf_matrix)[0]

        best_idx = np.argmax(sims)
        best_score = sims[best_idx]

        if best_score > 0.25:  # threshold for confidence
            st.success(f"**Answer:** {df.loc[best_idx, 'answer']}")
            st.caption(f"(Matched FAQ: '{df.loc[best_idx, 'question']}') – score={best_score:.2f}")
        else:
            st.warning("Sorry, I couldn't find a good match. Please rephrase your question.")
