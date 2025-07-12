import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ipywidgets as widgets
from IPython.display import display
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

# Step 5: Create chatbot interface with ipywidgets (using observe instead of on_submit)
def chatbot_widget():
    text_input = widgets.Text(
        value='',
        placeholder='Type your question related to Kaizen...',
        description='👨‍🏭 You:',
        layout=widgets.Layout(width='100%')
    )

    output = widgets.Output()
    display(text_input, output)

    # Ensure the widget does not trigger multiple times while typing
    text_input.continuous_update = False

    def handle_submit(change):
        user_query = change['new'].strip()
        if user_query:
            response = get_kaizen_response(user_query)
            with output:
                print(f"\n👨‍🏭 You: {user_query}")
                print(f"🤖 IndustryKaizen: {response}\n")
            text_input.value = ''  # Clear input

    # Attach the observer
    text_input.observe(handle_submit, names='value')

# Step 6: Run chatbot
chatbot_widget()
