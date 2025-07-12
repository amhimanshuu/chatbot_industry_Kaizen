import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Step 1: Load CSV without headers and assign custom column names
df = pd.read_csv("industrykaizen_qa.csv", header=None, names=["Problem_Description", "Kaizen_Solution"])

# Step 2: Extract questions and answers
questions = df["Problem_Description"].astype(str).tolist()
answers = df["Kaizen_Solution"].astype(str).tolist()

# Step 3: Create TF-IDF vectorizer and fit it on the questions
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

# Step 4: Define response function
def get_kaizen_response(user_input):
    user_input_clean = re.sub(r'[^\w\s]', '', user_input.lower())
    user_vector = vectorizer.transform([user_input_clean])
    similarity = cosine_similarity(user_vector, question_vectors)
    index = similarity.argmax()

    if similarity[0][index] > 0.2:  # Confidence threshold
        return answers[index]
    else:
        return "🤖 I'm sorry, I couldn't understand that. Please ask a Kaizen-related question."

# Step 5: Build Streamlit UI (replicating the chatbot experience)
st.set_page_config(page_title="IndustryKaizen Chatbot", layout="centered")
st.title("🤖 IndustryKaizen Chatbot")
st.markdown("Ask your Kaizen-related question below:")

# Persistent chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Text input from user
user_query = st.text_input("👨‍🏭 You:", placeholder="Type your Kaizen-related question and press Enter")

# If the user entered a query
if user_query:
    response = get_kaizen_response(user_query)
    st.session_state.history.append(("👨‍🏭 You", user_query))
    st.session_state.history.append(("🤖 IndustryKaizen", response))

# Display full conversation history
for sender, message in st.session_state.history:
    if sender == "👨‍🏭 You":
        st.markdown(f"**{sender}:** {message}")
    else:
        st.success(f"{sender}: {message}")
