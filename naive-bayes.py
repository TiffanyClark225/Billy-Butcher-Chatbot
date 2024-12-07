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
import random

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Load intents JSON
with open('intense.json') as file:
    intents = json.load(file)

lemmatizer = WordNetLemmatizer()

# Prepare data
corpus = []  # Stores all patterns
labels = []  # Stores corresponding tags (intents)

# Goes through the intents
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

# Initialize and train Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Predict intents for the test set
y_pred = classifier.predict(X_test)

# Print evaluation metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Function to classify user input intent
def classify_intent(user_input):
    # Preprocess user input
    tokens = nltk.word_tokenize(user_input)
    lemmatized = [lemmatizer.lemmatize(word.lower()) for word in tokens]
    processed_input = ' '.join(lemmatized)
    
    # Convert to TF-IDF features
    features = vectorizer.transform([processed_input]).toarray()
    
    # Predict intent
    predicted_tag = classifier.predict(features)[0]
    return predicted_tag


# Function to get a response based on the predicted intent
def get_response(predicted_tag):
    for intent in intents['intents']:
        if intent['tag'] == predicted_tag:
            return random.choice(intent['responses'])
    return "Sorry, I didn’t catch that."

# Test chatbot interaction
while True:
    print("Chatbot is up! Type 'exit' to quit.")
    user_input = input("You: ") # user input
    if user_input.lower() == "exit":
        print("Chatbot: Bye!")
        break
    predicted_tag = classify_intent(user_input)
    response = get_response(predicted_tag)
    print("Chatbot:", response) # chatbot's reponse
