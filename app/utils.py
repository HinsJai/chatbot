import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import random
import pickle
import json
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

from keras.models import load_model
from typing import List

ignore_symbols = ["?", "!", ".", ","]
ERROR_THRESHOLD = 0.25


def clean_up_sentense(sentence: str) -> list:

    lemmatizer = WordNetLemmatizer()

    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [
        lemmatizer.lemmatize(word)
        for word in sentence_words
        if word not in ignore_symbols
    ]

    return sentence_words


def bag_of_words(sentence: str) -> np.ndarray:
    words: List[str] = pickle.load(open("../model/words.pkl", "rb"))

    sentense_words: List[str] = clean_up_sentense(sentence)
    bag: List[int] = [0] * len(words)

    for w in sentense_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1

    return np.array(bag)


def predict_class(sentence) -> list:
    classes = pickle.load(open("../model/classes.pkl", "rb"))
    model = load_model("../model/chatbot_model.keras")

    bag_of_word = bag_of_words(sentence)
    res = model.predict(np.array([bag_of_word]))[0]

    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []

    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    print(classes[r[0]])
    return return_list


def get_response(intents_list):
    intents_json = json.load(open("../model/intents.json"))
    #default resposne tag is first intent
    tag = intents_list[0]["intent"]
    list_of_intents = intents_json["intents"]

    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break

    return result
