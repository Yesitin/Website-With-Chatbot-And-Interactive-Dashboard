import os 

# suppressing TF logging messages; '3' to only show errors
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import random
import pickle
import json

import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

# tokenizes and lemmatizes input
def clean_up_sentence(sentence):
    lemmatizer = WordNetLemmatizer()

    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = (lemmatizer.lemmatize(word) for word in sentence_words)

    return sentence_words

# converts cleaned sentence into bag-of-words vector based on chatbot's vocabulary -> input for the model
def bag_of_words(sentence):
    script_dir = os.path.dirname(os.path.abspath(__file__))  # setting the path
    words_pkl_file = os.path.join(script_dir, "..", "Flask Application", "model", "words.pkl")
    words = pickle.load(open(words_pkl_file, "rb"))

    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)

    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
                
    return np.array(bag)

# uses the trained model to predict intent of input sentence and returns list of intents with probabilities above a certain threshold
def predict_class(sentence):
    script_dir = os.path.dirname(os.path.abspath(__file__))  # setting the path
    classes_pkl_file = os.path.join(script_dir, "..", "Flask Application", "model", "classes.pkl")
    classes = pickle.load(open(classes_pkl_file, "rb"))

    chatbot_model_file = os.path.join(script_dir, "..", "Flask Application", "model", "chatbot_model.keras")
    model = load_model(chatbot_model_file)

    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25

    results = [[i,r ] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse = True)

    return_list = []

    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})

    return return_list

# takes predicted intent and fetches corresponding response from intents json file.
def get_response(intents_list):
    script_dir = os.path.dirname(os.path.abspath(__file__))  # setting the path
    intents_file = os.path.join(script_dir, "..", "Flask Application", "model", "intents.json")
    intents_json = json.load(open(intents_file))

    tag = intents_list[0]["intent"]
    list_of_intents = intents_json["intents"]

    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    
    return result
        


