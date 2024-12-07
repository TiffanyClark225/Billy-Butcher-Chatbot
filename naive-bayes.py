# Author: Tiffany Clark
# This file is a naive-bayes baseline model for our AI Chatbot

# Imports
import json
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from nltk.stem import WordNetLemmatizer

# Initialize lemmatizer and download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Load intents JSON
with open('intense.json') as file:
    intents = json.load(file)

lemmatizer = WordNetLemmatizer()

# Prepare data
corpus = []  # Stores all patterns
labels = []  # Stores corresponding tags (intents)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        tokens = nltk.word_tokenize(pattern)
        lemmatized = [lemmatizer.lemmatize(word.lower()) for word in tokens]
        corpus.append(' '.join(lemmatized))  # Preprocessed pattern
        labels.append(intent['tag'])         # Corresponding intent tag

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer(max_features=500)  # Limit the feature size for simplicity
X = vectorizer.fit_transform(corpus).toarray()  # TF-IDF feature matrix
y = labels                                       # Corresponding tags

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Predict intents for the test set
y_pred = classifier.predict(X_test)

# Print evaluation metrics
print("Overall Model Accuracy on Test Set:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Testing Suite
test_suite = [
    {"input": "Who are you?", "expected_tag": "Name"},
    {"input": "Do you trust people?", "expected_tag": "Trust_and_Loyalty"},
    {"input": "Why are you sarcastic?", "expected_tag": "Cynicism_and_Humor"},
    {"input": "Why do you hate Supes?", "expected_tag": "Rage_and_Passion"},
    {"input": "What do you think about Homelander?", "expected_tag": "Knowledge of Homelander"},
    {"input": "Tell me about Black Noir.", "expected_tag": "Knowledge of Black Noir"},
    {"input": "What do you know about The Deep?", "expected_tag": "Knowledge of The Deep"},
    {"input": "What do you think about A-Train?", "expected_tag": "Knowledge of A-Train"},
    {"input": "What do you know about Starlight?", "expected_tag": "Knowledge of Starlight"},
    {"input": "Who is Becca?", "expected_tag": "Knowledge of Becca"},
    {"input": "What do you think about Hughie?", "expected_tag": "Knowledge of Hughie"},
    {"input": "Tell me about Ryan.", "expected_tag": "Knowledge of Ryan"},
    {"input": "What do you know about Kimiko?", "expected_tag": "Knowledge of Kimiko"},
    {"input": "What do you know about Frenchie?", "expected_tag": "Knowledge of Frenchie"},
    {"input": "Tell me about MM.", "expected_tag": "Knowledge of MM"},
]

# Function to classify user input
def classify_intent(user_input):
    tokens = nltk.word_tokenize(user_input)
    lemmatized = [lemmatizer.lemmatize(word.lower()) for word in tokens]
    processed_input = ' '.join(lemmatized)
    features = vectorizer.transform([processed_input]).toarray()
    return classifier.predict(features)[0]

# Run the test suite
print("\nTesting Suite Results:\n")
total_tests = len(test_suite)
correct_predictions = 0

for test in test_suite:
    user_input = test["input"]
    expected_tag = test["expected_tag"]
    predicted_tag = classify_intent(user_input)
    
    # Check if the prediction matches the expected tag
    is_correct = predicted_tag == expected_tag
    if is_correct:
        correct_predictions += 1

    # Print results for each test case
    print(f"User Input: {user_input}")
    print(f"Expected Tag: {expected_tag}")
    print(f"Predicted Tag: {predicted_tag}")
    print(f"Correct Prediction: {is_correct}")
    print("-" * 50)

# Calculate and print accuracy for the test suite
accuracy = (correct_predictions / total_tests) * 100
print(f"Test Suite Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_tests} correct predictions)")
