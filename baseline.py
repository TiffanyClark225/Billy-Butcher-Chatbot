# Author:  Charles Oakes
# This file is the baseline model for our AI Chatbot
# This script implements a Markov chain-based chatbot that generates responses in Billy Butcher's style.


# Imports
import random
import re

# Corpus Processing
# Processes the input text corpus by lowercasing, tokenizing, and separating punctuation into individual tokens
# Returns list of processed words from the corpus
def process_corpus(corpusFile): 
    with open(corpusFile, 'r') as file:
        corpus = file.read()

    # Lowercase and tokenize, keeping punctuation as separate tokens
    corpus = corpus.lower()
    corpus = re.sub(r'([.,!?;])', r' \1 ', corpus)  # Add spaces around punctuation
    corpusWords = corpus.split()

    return corpusWords  #, len(corpusWords)  <- Used for testing purposes

# Markov Chain Building
# Builds an n-gram Markov chain from the processed corpus
def build_chains(corpusWords, n=3):
    chain = {}
    total_words = len(corpusWords)

    # Build n-grams and map to possible next words
    for i in range(len(corpusWords) - n):
        key = tuple(corpusWords[i: i + n])
        next_word = corpusWords[i + n]

        if key not in chain:
            chain[key] = []
        chain[key].append(next_word)

    # Laplace Smoothing (add 1 to every word count)
    for key in chain:
        # Normalize: distribute "extra" counts to unseen words
        total_count = len(chain[key]) + len(set(corpusWords))
        chain[key] = [word for word in chain[key]]  # Adjust as needed

    return chain

# Generats a response using the Markov chain based on the given seen or random start
def response(chain, seed=None, length = 50, n = 6):         #the part where billy actually speaks!! (yay, maybe)

    if seed is None:
        seed = random.choice(list(chain.keys()))  # random seed, Butcher will ramble incoherently

    response = list(seed)

    # Generate words until the desired length is reached
    for _ in range(length - len(seed)):
        key = tuple(response[-n:])  # focuses the last n words for the key
        if key in chain:
            next_word = random.choice(chain[key])       # tries to randomly select the next word from the chain
            response.append(next_word)
        else:
            # Fallback: Try smaller n-gram (backoff mechanism)
            if n > 1:
                return response(chain, seed=tuple(response[-(n-1):]), length=length, n=n-1)
            break
    # Join the response into a single string and return it
    return " ".join(response)

# Processes user input to find a suitable seed for generating a response
def handle_input(usr_input, chain, n=3):        #converts user input to stuff Butcherbot can read

    words = usr_input.lower().split()

    if len(words) < n:
        return None  # input not big enough to create the n(tri)gram

    seed = tuple(words[-n:])  #focuses the last n words to find a response (prob a really bad way to do this)

    if seed in chain:
        return seed  # good seed
    return None  # no good seed

# Main Chatbot Interaction
if __name__ == "__main__":
    # Load dialogue corpus and process it into words
    # CHANGE THIS TO MATCH YOUR FILE PATH
    corpusFile = "/workspaces/Billy-Butcher-Chatbot/Dialogue/dialogue.txt"

    words = process_corpus(corpusFile)

    """Testing stuff"""
    #wordcount = len(words)
    #print(f"Processed corpus size: {word_count} words")

    
    #print(words)
    #print("\n\n")
    """Testing stuff"""

    n = 6  # ngram count
    markov_chain = build_chains(words, n = n)

    """Testing stuff"""
    #print(markov_chain)
    """Testing stuff"""


    # Chatbot Interaction loop
    print("Butcher: Oi. Let's have ourselves a little chat, eh? Or type 'exit' to bugger off.")
    while True:
        user_input = input("You: ") # Get user input
        if user_input.lower() == "exit": # Exit condition
            print("Butcher: Beat it.")
            break

        # Generates response
        seed = handle_input(user_input, markov_chain, n = n)
        if seed:
            botResponse = response(markov_chain, seed=seed, n = n)
        else:
            botResponse = response(markov_chain, n = n)  # Random response if no seed match

        print(f"Butcher: {botResponse}") # Prints the chatbot's response
