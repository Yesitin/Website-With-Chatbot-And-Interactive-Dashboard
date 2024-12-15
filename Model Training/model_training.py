import os 

# suppressing TF logging messages; '3' to only show errors
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import random
import pickle
import json

import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import SGD

nltk.download('punkt_tab')


# initializing and load intents file 
lemmatizer = WordNetLemmatizer()

script_dir = os.path.dirname(os.path.abspath(__file__))  # setting the path
intents_file = os.path.join(script_dir, "intents.json")
intents = json.load(open(intents_file))



words = []
classes = []
documents = []
ignore_symbols = ["?", "!", ".", ","]

# process each pattern in the intents
for intent in intents["intents"]:
    for pattern in intent["patterns"]:

        # tokenizing each pattern into indivudial words
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)                         # adding words to the list
        documents.append((word_list, intent["tag"]))    # appending document data
        if intent["tag"] not in classes:
            classes.append(intent["tag"])               # adding unique tags to classes


# lemmatizing and removing ignore_letters from the list of words
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_symbols]
words = sorted(set(words))

# sorting and saving classes 
classes = sorted(set(classes))



words_pkl_path = os.path.join(script_dir, "..", "Flask Application", "model", "words.pkl")
classes_pkl_path = os.path.join(script_dir, "..",  "Flask Application", "model", "classes.pkl")

pickle.dump(words, open(words_pkl_path, "wb"))
pickle.dump(classes, open(classes_pkl_path, "wb"))

# preparing training data
training = []
output_empty = [0] * len(classes)   # creating zero vector for the output


for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns if word not in ignore_symbols]

    # creating bag-of-words (binary vector) where each position corresponds to a word in the vocabulary
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    # creating output row with a 1 at the index of the class/tag
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

# training/test data split
random.shuffle(training)
training = np.array(training, dtype= object)

train_x = np.array([item[0] for item in training])
train_y = np.array([item[1] for item in training])

# building neural network model
model = Sequential()
model.add(Input(shape=(len(train_x[0]),)))
#model.add(Dense(128, activation = "relu"))     # input layer with 128 neurons
model.add(Dropout(0.5))                         # Dropout layer to prevent overfitting
model.add(Dense(units=64, activation = "relu")) # Hidden layer with 64 neurons
model.add(Dropout(0.5))                         # Dropout layer
model.add(Dense(len(train_y[0]), activation = "softmax"))   # Output layer with a neuron for each class

# compiling model with sgd optimizer and train it 
sgd = SGD(learning_rate = 0.01, weight_decay = 1e-6, momentum = 0.9, nesterov = True)
model.compile(loss = "categorical_crossentropy", optimizer = sgd, metrics = ["accuracy"])

model.fit(np.array(train_x), np.array(train_y), epochs = 200, batch_size = 5, verbose = 1)



# saving the model

model_path = os.path.join(script_dir, "..", "Flask Application", "model", "chatbot_model.keras")  # setting the path

model.save(model_path)