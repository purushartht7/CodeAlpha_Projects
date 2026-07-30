"""
CodeAlpha - Task 2: Chatbot for FAQs
Author: Ashutosh

A simple FAQ chatbot that matches a user's question to the closest
FAQ using TF-IDF + cosine similarity, and returns the best answer.

Run locally with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Page setup ----------
st.set_page_config(page_title="FAQ Chatbot", page_icon="💬", layout="centered")
st.title("💬 FAQ Chatbot")
st.caption("Ask me anything about the internship process — I'll match it to the closest known FAQ.")

# ---------- Load FAQ data ----------
@st.cache_data
def load_faqs():
    return pd.read_csv("faqs.csv")

faqs = load_faqs()

# ---------- Preprocessing ----------
def clean_text(text: str) -> str:
    """Lowercase, strip punctuation/extra whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

faqs["clean_question"] = faqs["question"].apply(clean_text)

# ---------- Build the TF-IDF matching model ----------
@st.cache_resource
def build_vectorizer(questions):
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(questions)
    return vectorizer, matrix

vectorizer, faq_matrix = build_vectorizer(faqs["clean_question"])

def get_best_answer(user_question: str, threshold: float = 0.25):
    cleaned = clean_text(user_question)
    user_vec = vectorizer.transform([cleaned])
    similarities = cosine_similarity(user_vec, faq_matrix).flatten()
    best_idx = similarities.argmax()
    best_score = similarities[best_idx]

    if best_score < threshold:
        return ("I'm not confident I know the answer to that. "
                "Try rephrasing, or ask something about the CodeAlpha internship process."), best_score
    return faqs.iloc[best_idx]["answer"], best_score

# ---------- Chat UI ----------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask me a question about the CodeAlpha internship, and I'll do my best to answer."}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Type your question here...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    answer, score = get_best_answer(user_input)
    with st.chat_message("assistant"):
        st.write(answer)
        st.caption(f"Match confidence: {score:.2f}")

    st.session_state.messages.append({"role": "assistant", "content": answer})

with st.expander("📋 View all FAQs in the knowledge base"):
    st.dataframe(faqs[["question", "answer"]], hide_index=True, use_container_width=True)

st.divider()
st.caption("Built with Streamlit + scikit-learn (TF-IDF & cosine similarity) · CodeAlpha AI Internship")
