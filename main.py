# Import Libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
import pickle
import json
import random
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer


lemm = WordNetLemmatizer()

# Loading the JSON
with open('intents.json') as file:
    data = json.load(file)

# Creating empty list and Ignoring some letters
words = []
labels = []
docs = []
ignore_letters = ['?', '!', '.', ',']



# Tokenize words
for intent in data['intents']:
    for patterns in intent['patterns']:
        words_list = nltk.word_tokenize(patterns)
        words.extend(words_list)
        docs.append((words_list, intent['tag']))
        if intent['tag'] not in labels:
            labels.append(intent['tag'])


# Stemming the words
words = [lemm.lemmatize(w.lower()) for w in words if w != ignore_letters]
words = sorted(list(set(words)))
labels = sorted(list(set(labels)))

# Save the List
pickle.dump(words, open('words.pk1', 'wb'))
pickle.dump(labels, open('labels.pk1', 'wb'))


training = []
output = []

out_empty = [0 for _ in range(len(labels))]

# Turn the words into numbers
for doc in docs:
    bag = []
    words_pattern = doc[0]
    words_pattern = [lemm.lemmatize(w.lower()) for w in words_pattern]

    for w in words:
        if w in words_pattern:
            bag.append(1)
        else:
            bag.append(0)

    output_rows = list(out_empty)
    output_rows[labels.index(doc[1])] = 1

    training.append([bag, output_rows])

# Preparing for model
random.shuffle(training)
training = np.array(training)


train_X = list(training[:, 0])
train_y = list(training[:, 1])


# Set up the model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_X[0]), ), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
history = model.fit(np.array(train_X), np.array(train_y), batch_size=5, epochs=200, verbose=1)
accuracy = model.evaluate(X_test, y_test)
print(accuracy)

# Save the Model
model.save('chat model.h5', history)






