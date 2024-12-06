import random
import json
import pickle
import numpy as np
import nltk
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

# Load intents JSON
intents = json.loads(open("intense.json").read())

# Prepare data
words = []
classes = []
documents = []
ignore_letters = ["?", "!", ".", ",", ":", ";", "'"]

for intent in intents['intents']: #process words and classes, so that it can be used in a neural network environment.
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern) 
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters] #sort and lemmatize (group by similarity, etc.) words, 
words = sorted(set(words))

pickle.dump(words, open('words.pkl', 'wb')) #create new container for classes, words.
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)

for document in documents: #creates bag of words, (i.e. corpus)
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1 #seperating processed outputs from input corpus
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

#ALL OF THE HYPERPARAMETERS BELOW WERE SELECTED VIA VARIOUS TRIAL-AND-ERROR PROCESSES
model = Sequential()  #neural network framework with exactly one input and output
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))  #create a fully connected neural network (Dense) with 128 hidden units (128) and we ensured that 
model.add(Dropout(0.5))                                                   #the input, the training data, has it's size/shape correctly mapped( len(train)). reLU as the activation function 
model.add(Dense(64, activation='relu'))                                   #primarily for speed. 
model.add(Dropout(0.5))                                                   #dropout to .5 helps combat overfitting, especially in our fairly specific feature vectors
model.add(Dense(len(train_y[0]), activation='softmax'))                   #softmax is used to develop a probability distribution over our outputs. 



sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)                 #SGD with an adequately low learning rate (.01) that gets gradually smaller (decay = 1e-6). This hurts 
                                                                                       #runtime complexity but ensure a converging gradient. This is counteracted by a high momentum value that
                                                                                       #will very rapidly approach the universal minimum, ignore local minima in the loss function. nesterov (NAG) 
                                                                                       #looks ahead, so the momentum hyperparameter will work as intended, efficiently.

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])    #categorical cross entropy is good for categorical classification of one-hot encoded labels; our exact situation.
                                                                                       #we use SGD, as seen above
                                                                                       #we monitor accuracy during training, minimizing loss relating to that.

model.fit(np.array(train_x), np.array(train_y), epochs=500, batch_size=5, verbose=1)   #train the model. After ensuring our features and labels are correctly formatted, train over 500 (arbitrary, but important) epochs.
                                                                                       #updated weights every 5 samples
                                                                                       #verbose being toggled outputs progress bars for each epoch of training.
model.save("billy-buster.keras") #

print("Training complete!")


