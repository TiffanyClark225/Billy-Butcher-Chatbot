# Import your chatbot code
import json
import pickle
import numpy as np
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
import nltk
from difflib import SequenceMatcher

# Load chatbot data and model
lemmatizer = WordNetLemmatizer()
intents = json.loads(open("intense.json").read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('billy-buster.keras')

# Define the chatbot response functions (copied from your main.py)
def clean_up_sentences(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bagw(sentence):
    sentence_words = clean_up_sentences(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bagw(sentence)
    res = model.predict(np.array([bow]), verbose=0)[0]
    ERROR_THRESHOLD = 0.5  # Increase threshold
    results = [{"intent": classes[i], "probability": float(r)} for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x["probability"], reverse=True)
    return results


def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            return np.random.choice(i['responses'])
    return "Sorry, I didn't understand that. Please try again!"

def chatbot_response(message):
    intents_list = predict_class(message)
    if intents_list:
        return get_response(intents_list, intents)
    else:
        return "Sorry, I didn't understand that. Please try again!"

# Function to check for fuzzy matching
def fuzzy_match(bot_response, expected_responses, threshold=0.7):
    for expected in expected_responses:
        similarity = SequenceMatcher(None, bot_response, expected).ratio()
        if similarity >= threshold:
            return True
    return False

def normalize_text(text):
    """Normalize text by stripping whitespace and converting to lowercase."""
    return text.strip().lower()

# Define the Test Suite with Expected Tags
test_suite = [
    {
        "input": "Who are you?",
        "expected_tag": "Name",
        "expected_responses": ["Butcher", "Billy Butcher", "They call me Butcher", "Butcher. Billy Butcher."]
    },
    {
        "input": "What’s up?",
        "expected_tag": "Concern",
        "expected_responses": ["Oi, don't panic. We'll sort this out, mate.", "Well, you can kindly fuck off then."]
    },
    {
        "input": "Do you trust me?",
        "expected_tag": "Trust_and_Loyalty",
        "expected_responses": ["I trust no one, especially not the bloody Supes."]
    },
    {
        "input": "What’s your name?",
        "expected_tag": "Name",
        "expected_responses": ["Butcher", "Billy Butcher", "They call me Butcher", "Butcher. Billy Butcher."]
    },
    {
        "input": "How are you?",
        "expected_tag": "Concern",
        "expected_responses": ["Oi, don’t worry about me. I’m bloody great.", "It’s a sh*tshow, but we’ve seen worse, haven’t we?"]
    },
    {
        "input": "What’s the plan?",
        "expected_tag": "Leadership_and_Strategy",
        "expected_responses": ["We’re sniffing down a sh*t sandwich the size of Watergate."]
    },
    {
        "input": "Do Supes deserve trust?",
        "expected_tag": "Morality_and_Power",
        "expected_responses": ["With great power comes the absolute certainty that you'll turn into a right cunt."]
    },
    {
        "input": "What should I do?",
        "expected_tag": "Action_and_Persuasion",
        "expected_responses": ["Depends. You got anything to prove you’re not a muppet?"]
    },
    {
        "input": "asdfg",
        "expected_tag": None,
        "expected_responses": ["Sorry, mate. I didn’t catch that.", "Sorry, I didn't understand that. Please try again!"]
    },
    {
        "input": "Tell me about Supes.",
        "expected_tag": "Morality_and_Power",
        "expected_responses": ["Supes are all the same. Every bloody one of them."]
    }
]


# Test the Chatbot
def run_test_suite_with_grading():
    print("Running Chatbot Test Suite...\n")
    total_tests = len(test_suite)
    passed_tests = 0
    manual_review = []

    for test_data in test_suite:  # Iterate directly over the list
        user_input = test_data["input"]
        expected_responses = test_data["expected_responses"]
        expected_tag = test_data["expected_tag"]

        print(f"Input: {user_input}")
        predicted_intents = predict_class(user_input)
        bot_response = chatbot_response(user_input)

        if predicted_intents:
            predicted_tag = predicted_intents[0]["intent"]
        else:
            predicted_tag = None

        print(f"Predicted Tag: {predicted_tag}")
        print(f"Expected Tag: {expected_tag}")
        print(f"Bot Response: {bot_response}")
        print(f"Expected Responses: {expected_responses}\n")

        # Check for tag match
        if predicted_tag == expected_tag:
            # Check for response match
            normalized_bot_response = normalize_text(bot_response)
            normalized_expected_responses = [normalize_text(response) for response in expected_responses]
            if normalized_bot_response in normalized_expected_responses:
                print("✅ Test Passed")
                passed_tests += 1
            elif fuzzy_match(bot_response, expected_responses):
                print("🔄 Partial Match: Intent Correct, Response Needs Review")
                manual_review.append((user_input, predicted_tag, expected_tag, bot_response, expected_responses))
            else:
                print("❌ Test Failed: Incorrect Response")
                manual_review.append((user_input, predicted_tag, expected_tag, bot_response, expected_responses))
        else:
            print("❌ Test Failed: Incorrect Tag")
            manual_review.append((user_input, predicted_tag, expected_tag, bot_response, expected_responses))
        print("-" * 50)

    # Display grading metrics
    intent_accuracy = (passed_tests / total_tests) * 100
    print(f"\nTest Suite Results: {passed_tests}/{total_tests} tests passed.")
    print(f"Intent Accuracy: {intent_accuracy:.2f}%")
    print(f"Manual Reviews Needed: {len(manual_review)}\n")
    if manual_review:
        print("Manual Review Responses:")
        for review in manual_review:
            print(f"Input: {review[0]}")
            print(f"Predicted Tag: {review[1]}")
            print(f"Expected Tag: {review[2]}")
            print(f"Bot Response: {review[3]}")
            print(f"Expected Responses: {review[4]}")
            print("-" * 50)


# Run the updated test suite
if __name__ == "__main__":
    run_test_suite_with_grading()
