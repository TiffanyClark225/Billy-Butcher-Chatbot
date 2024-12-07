## Billy-Butcher-Chatbot

This is our final Machine Learning (CS460G) project. It is a AI chatbot based on the tone and personality of Billy Butcher in Amazon's series The Boys. We have a Marklov-chain based chatbot and Naive-Bayes chatbot for our baseline models, and used Neural Network tensorflow for our final chatbot.


# Authors:
Charles Oakes (12608196)
- Researched different implementation plans for chatbots
- Created Baseline Model (baseline.py)
- Expanded tensorflow intense.json file

Dayne Freudenberg (12576580)
- Researched python libraries and potential implementation techniques
- Created the initial intense.json file for training
- Created main.py

Tiffany Clark (12556043)
- Wrote Automated_Parsing.ipynb to parse Butcher's dialogue from episode scripts
- Wrote the final tensorflow intense.json file with more tags, inputs, and responses
- Created testing suite by writing User Input.txt and Response.txt with potential questions to ask the chatbot and Butcher's responses
- Wrote training.py script to go through a mini testing suite instead of manually having to input testing questions
- Wrote Naive-Bayes.py for baseline model

Logan Hester (912111755)
- Expanded Dayne's initial intense.json file by adding more responses from the dialogue.txt file Tiffany created
- Wrote potential questions for testing that Tiffany implemented in the testing suite
- Conducted comparative analysis of baseline models, helping to identify limitations and guiding improvements for the final baseline chatbot implementation

# Files
- Automated_Parsing.ipynb = parses Billy Butcher's dialogue from different scripts online and adds to dialogue.txt
- Dialogue.txt = Lines of dialogue from Billy Butcher to use for intense.json file and testing
- User Input.txt = Questions created to help test the chatbot and evaluate its responses
- Response.txt = Potential responses (Butcher's dialogue) to help test and evaluate the chatbot's responses
- Markov-Chain.py = Markov Chain-based chatbot code for our baseline model
- intense.json = intents file to train the tensorflow chatbot with tags, inputs, and potential responses
- main.py = file that runs the tensorflow chatbot
- naive-bayes.py = Naive-Bayes based baseline model
- requirements = list of requirements to run our code (installs and imports)
- NN_testing.py = file made that automatted runs and asks the tensorflow chatbot questions to test and evaluate how it does
- NN_training.py = file that trains the tensorflow chatbot based on the intense.json file

# How to Run the Code
1. Type on terminal: pip install -r requirements.txt
2. To try our Naive-Bayes Model type on terminal: python naive-bayes.py
3. To try our Markov-Chains Model type on terminal: python Markov-Chains.py
4. To train our final model type on terminal: python NN_training.py
5. To test final model type on terminal: python NN_testing.py
6. To try our final model type on terminal: python main.py
