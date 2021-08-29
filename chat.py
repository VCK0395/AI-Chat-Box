# Import Libraries
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# Loading necessary files
lemmatizer = WordNetLemmatizer()
intent = json.loads(open('intents.json').read())
word = pickle.load((open('words.pk1', 'rb')))
labels = pickle.load((open('labels.pk1', 'rb')))
model = load_model('chat model.h5')

class Chat:
    # Clean the sentence
    def Clean_sentence(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [lemmatizer.lemmatize(words) for words in sentence_words]
        return sentence_words

    # Making 0 and 1 for each words
    def Bag_of_words(self, sentence):
        sentence_words = self.Clean_sentence(sentence)
        bag = [0] * len(word)
        for w in sentence_words:
            for i in enumerate(word):
                if word == w:
                    bag[i] = 1
        return np.array(bag)

    # Making the Chat predict
    def Predict_class(self, sentence):
        bow = self.Bag_of_words(sentence)
        res = model.predict(np.array([bow]))[0]
        result = [[i, r] for i, r in enumerate(res) if r > 0.25]
        result.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in result:
            return_list.append({'intents': labels[r[0]], 'probability': str(r[1])})
        return return_list

    def GetResponse(self, ints, intents_json):
        tag = ints[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if (i['tag'] == tag):
                result = random.choice(i['responses'])
                break
        return result

    def GetResponse(self, ints, intents_json):
        tag = ints[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if (i['tag'] == tag):
                result = random.choice(i['responses'])
                break
        return result

    def chatbot_response(self, text):
        ints = self.Predict_class(text, model)
        res = self.GetResponse(ints, intent)
        return res

ChatBox = Chat()

print(ChatBox)





print("Go! Bot is running")

