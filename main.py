# Authors: Dayne and Tiffany (few changes)
# This file runs the TensorFlow AI chatbot and includes optional Hugging Face integration for generating responses.

# Imports
import random
import json
import pickle
import numpy as np
import nltk
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
from transformers import pipeline
from tf_keras.models import Sequential
from tf_keras.layers import Dense, Activation, Dropout
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, TextDataset



# Disable GPU usage to ensure the chatbot runs only on the CPU
# Added this because of errors I was getting
# Tiffany Clark
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Optional Hugging Face Pipeline
# Uncomment these lines to enable Hugging Face for backup response generation.
# Hugging Face provides an additional method to generate responses if TensorFlow predictions are uncertain.
# Tiffany Clark
# Hugging Face pipeline setup
#hugging_face_model = pipeline("text-generation", model="distilgpt2")

# Dayne
# Loading files
lemmatizer = WordNetLemmatizer()
intents = json.loads(open("intense.json").read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('billy-buster.keras')

# Dayne
#natural word processing of user inputs, using the nltk python library 
#This tokenizes and lemmatizes the words
def clean_up_sentences(sentence): 
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# Dayne
#creates the bag of words (corpus) for user inputs, and enumerates them accordingly. 
def bagw(sentence):
    sentence_words = clean_up_sentences(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Dayne
#Use the model to predict the class of the user input. 
#function returns a list of all the probabilities for each corressponding class/output. 
def predict_class(sentence):
    bow = bagw(sentence)
    res = model.predict(np.array([bow]), verbose=0)[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

# Dayne
#takes the probability given by the function above to randomly select an output of the highest probability class.
#returns that response.
def get_tensorflow_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return "Sorry, I didn't understand that."

# Tiffany Clark
"""def get_huggingface_response(message):
    try:
        prompt = f"You are Billy Butcher from 'The Boys.' Respond to the following in his sarcastic, aggressive tone: {message}"
        response = hugging_face_model(message, max_length=50, num_return_sequences=1)
        return response[0]["generated_text"]
    except Exception as e:
        print(f"Error with Hugging Face: {e}")
        return "Sorry, I couldn't process that with Hugging Face."

def get_response(intents_list, intents_json, message):
    if intents_list and float(intents_list[0]['probability']) > 0.8:
        return get_tensorflow_response(intents_list, intents_json)
    else:
        return get_huggingface_response(message)"""

print("Chatbot is up! Type exit to end chat.")

# Dayne
# Continuously listen for user input until the user types "exit"
while True:
    message = input("> ")
    if message.lower() == "exit":
        break
    # Predict the intent of the user input
    intents_list = predict_class(message)
    if intents_list:
        #response = get_response(intents_list, intents, message)
        response = get_tensorflow_response(intents_list, intents)
        print(response)
    else:
        print("Sorry, I didn't understand that. Please try again!")
