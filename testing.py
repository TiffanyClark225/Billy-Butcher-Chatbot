# Author: Tiffany Clark
# This file handles running the testing suite for the TensorFlow AI Chatbot

# Imports
import json
import pickle
import numpy as np
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
import nltk

# Load chatbot data and model
# Loads the intents, preprocessed vocab, class labels, and trained TensorFlow mode
lemmatizer = WordNetLemmatizer()
intents = json.loads(open("intense.json").read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('billy-buster.keras')

# Define chatbot response functions

# Tokenizes and lemmatizes the input sentence to prepare it for processing
def clean_up_sentences(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Converts a cleaned sentence into a bag of words representation for model input
def bagw(sentence):
    sentence_words = clean_up_sentences(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Predicts the most likely intent class based on the input sentence
def predict_class(sentence):
    bow = bagw(sentence)
    res = model.predict(np.array([bow]), verbose=0)[0]
    ERROR_THRESHOLD = 0.5  # Threshold for intent prediction
    results = [{"intent": classes[i], "probability": float(r)} for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x["probability"], reverse=True)
    return results

# Returns a response based on the predicted intent
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            return np.random.choice(i['responses'])
    # Fallback response 
    return "Sorry, mate. I didn’t catch that. Try again."

# Combines prediction and response generation to return the chatbot's reply to a user message
def chatbot_response(message):
    intents_list = predict_class(message)
    if intents_list:
        return get_response(intents_list, intents)
    # If not intent is found, fallback response is given
    else:
        return "Sorry, mate. I didn’t catch that. Try again." 

# Define the Test Suite
# List of test cases with input messages and their expected tags
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

# Run the Test Suite and evaluates the chatbot's intent prediction accuracy
def run_test_suite():
    print("Running Chatbot Test Suite...\n")
    total_tests = len(test_suite)
    passed_tests = 0
    failed_tests = []

    # Iterate through every test case
    for test_data in test_suite:
        user_input = test_data["input"]
        expected_tag = test_data["expected_tag"]

        print(f"Input: {user_input}")
        predicted_intents = predict_class(user_input)
        bot_response = chatbot_response(user_input)

        if predicted_intents:
            predicted_tag = predicted_intents[0]["intent"]
        else:
            predicted_tag = None

        # For each test case, print out predicted tag, expected tag, bot response, and whether it
        # correctly or incorrectly predicted the tag
        print(f"Predicted Tag: {predicted_tag}")
        print(f"Expected Tag: {expected_tag}")
        print(f"Bot Response: {bot_response}\n")

        if predicted_tag == expected_tag:
            print("✅ Correctly Predicted Tag")
            passed_tests += 1
        else:
            print("❌ Wrongly Predicted Tag")
            failed_tests.append({
                "input": user_input,
                "predicted_tag": predicted_tag,
                "expected_tag": expected_tag,
                "bot_response": bot_response
            })
        print("-" * 50)

    # Calculate accuracy - based on predicted vs. expected tag
    accuracy = (passed_tests / total_tests) * 100
    print(f"\nTest Suite Results: {passed_tests}/{total_tests} tests passed.")
    print(f"Accuracy: {accuracy:.2f}%")

    # Print failed tests for manual review
    if failed_tests:
        print("\nFailed in Predicting Tag but Need Manual Review")
        for failure in failed_tests:
            print(f"Input: {failure['input']}")
            print(f"Predicted Tag: {failure['predicted_tag']}")
            print(f"Expected Tag: {failure['expected_tag']}")
            print(f"Bot Response: {failure['bot_response']}")
            print("-" * 50)

if __name__ == "__main__":
    run_test_suite()
