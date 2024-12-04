import random

def process_corpus(corpusFile): # reads in the script file and tokenizes it for building Markov Chains

    with open(corpusFile, 'r') as file:
        corpus = file.read()

    corpus = corpus.lower()  
    corpus = corpus.replace("\n", " ").replace(".", " .").replace(",", " ,").replace("?", " ?").replace("!", " !")
    corpusWords = corpus.split()
    return corpusWords #, len(corpusWords)  <- Used for testing purposes


def build_chains(corpusWords, n=3):         #Builds markov chains using trigrams, could be changed by chainging the value of n tho

    chain = {}

    for i in range(len(corpusWords) - n):
        key = tuple(corpusWords[i: i + n])  # creates the n(tri)gram
        next_word = corpusWords[i + n]

        if key not in chain:
            chain[key] = []
        chain[key].append(next_word)  # adds to the list of possible next words

    return chain


def response(chain, seed=None, length = 50, n = 6):         #the part where billy actually speaks!! (yay, maybe)

    if seed is None:
        seed = random.choice(list(chain.keys()))  # random seed, Butcher will ramble incoherently

    response = list(seed)

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

    return " ".join(response)


def handle_input(usr_input, chain, n=3):        #converts user input to stuff Butcherbot can read

    words = usr_input.lower().split()

    if len(words) < n:
        return None  # input not big enough to create the n(tri)gram

    seed = tuple(words[-n:])  #focuses the last n words to find a response (prob a really bad way to do this)

    if seed in chain:
        return seed  # good seed
    return None  # no good seed


if __name__ == "__main__":
    corpusFile = "Billy-Butcher-Chatbot\Butcher_Compiled_Dialogue (1).txt"

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


    
    print("Butcher: Oi. Let's have ourselves a little chat, eh? Or type 'exit' to bugger off.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Butcher: Beat it.")
            break

        seed = handle_input(user_input, markov_chain, n = n)
        if seed:
            botResponse = response(markov_chain, seed=seed, n = n)
        else:
            botResponse = response(markov_chain, n = n)  # Random response if no seed match

        print(f"Butcher: {botResponse}")
