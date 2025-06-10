import json
import random
import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

DATASET_FILE = 'dataset.json'
MODEL_FILE = 'chatbot_model.pkl'
VECTORIZER_FILE = 'vectorizer.pkl'
LABEL_ENCODER_FILE = 'label_encoder.pkl'

def load_dataset(file_path=DATASET_FILE):
    if not os.path.exists(file_path):
        print(f"Error: Dataset file '{file_path}' not found.")
        return [], []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            questions = []
            answers = []
            for entry in data:
                q = entry.get('question', '').strip()
                ans_list = entry.get('answers', [])
                if q and ans_list:
                    questions.append(q)
                    answers.append(random.choice(ans_list).strip())
            return questions, answers
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON file '{file_path}': {e}")
        return [], []

def train_and_save_model():
    print("Loading dataset...")
    questions, answers = load_dataset()
    if not questions or not answers:
        print("Dataset is empty or invalid. Training aborted.")
        return

    print("Vectorizing questions...")
    vectorizer = CountVectorizer()
    x = vectorizer.fit_transform(questions)

    print("Encoding answers...")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(answers)

    print("Training classifier...")
    classifier = LogisticRegression(max_iter=200)
    classifier.fit(x, y)

    print("Saving model, vectorizer, and label encoder...")
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(classifier, f)
    with open(VECTORIZER_FILE, 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(LABEL_ENCODER_FILE, 'wb') as f:
        pickle.dump(label_encoder, f)

    print("Training completed and model saved.")

def load_model():
    if not (os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE) and os.path.exists(LABEL_ENCODER_FILE)):
        print("Model files missing. Train the model first.")
        return None, None, None
    with open(MODEL_FILE, 'rb') as f:
        classifier = pickle.load(f)
    with open(VECTORIZER_FILE, 'rb') as f:
        vectorizer = pickle.load(f)
    with open(LABEL_ENCODER_FILE, 'rb') as f:
        label_encoder = pickle.load(f)
    return classifier, vectorizer, label_encoder

def get_response(user_input, classifier, vectorizer, label_encoder):
    x_test = vectorizer.transform([user_input])
    y_pred = classifier.predict(x_test)
    answer = label_encoder.inverse_transform(y_pred)
    return answer[0]

def chatbot_loop():
    classifier, vectorizer, label_encoder = load_model()
    if classifier is None:
        print("Training model now...")
        train_and_save_model()
        classifier, vectorizer, label_encoder = load_model()
        if classifier is None:
            print("Failed to load model after training. Exiting.")
            return

    print("\nWelcome to the Skincare Chatbot!")
    print("Type your question and press Enter.")
    print("Type 'train' to retrain the model after dataset updates.")
    print("Type 'exit', 'quit', or 'bye' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            print("Please enter your question.")
            continue
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Chatbot: Goodbye! Take care of your skin!")
            break
        if user_input.lower() == 'train':
            print("Retraining model...")
            train_and_save_model()
            classifier, vectorizer, label_encoder = load_model()
            continue
        try:
            response = get_response(user_input, classifier, vectorizer, label_encoder)
            print(f"Chatbot: {response}")
        except Exception as e:
            print("Chatbot: Sorry, I didn't get that. Please try again.")

            # For debugging, uncomment the next line:
            # print(f"[Debug] {e}")
if __name__ == "__main__":
    chatbot_loop()
