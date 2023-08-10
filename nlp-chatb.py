from functools import lru_cache
import nltk
from json import load as json_load
from random import choice
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


with open("cybot_dataset.json","r") as file: dataset = json_load(file)
def response(prompt:str) -> str:
    global dataset
    for ele in dataset["intents"]:
        if prompt.capitalize() in ele["patterns"]:
            return choice(ele["patterns"])
    
    intents = dataset["intents"]
    patterns_corpus = [" ".join(intent["patterns"]) for intent in intents]
    vectorizer = CountVectorizer()
    x = vectorizer.fit_transform(patterns_corpus)
    nlp = spacy.load("en_core_web_sm")
    user_doc = nlp(prompt)
    user_vector = vectorizer.transform([" ".join([token.text for token in user_doc])])
    similarities = cosine_similarity(x, user_vector)
    intent_index = np.argmax(similarities)
    max_similarity = similarities[intent_index]
    if max_similarity > 0.25 or (1 - max_similarity) < 0.25:
        return choice(intents[intent_index]["responses"])

    @lru_cache(maxsize=16)
    def edit_dist(s1:str,s2:str,transpos:bool) -> int: return nltk.edit_distance(s1,s2,transpositions=transpos)
    max_overlap = 0
    matched_key = None
    for intent in dataset["intents"]:
        for pattern in intent["patterns"]:
            if edit_dist(pattern, prompt, True) > 5: continue
            overlap = len(set(pattern) & set(prompt))
            if overlap > max_overlap:
                max_overlap = overlap
                matched_key = intent["tag"]
    if matched_key:
        matched_intent = None
        for intent in dataset["intents"]:
            if intent["tag"] == matched_key:
                matched_intent = intent
                break
        if matched_intent:
            response = choice(matched_intent["responses"])
            return response
def get_intent(user_input):
    intents = dataset["intents"]
    patterns_corpus = [" ".join(intent["patterns"]) for intent in intents]
    vectorizer = CountVectorizer()
    x = vectorizer.fit_transform(patterns_corpus)
    nlp = spacy.load("en_core_web_sm")
    user_doc = nlp(user_input)
    user_vector = vectorizer.transform([" ".join([token.text for token in user_doc])])
    similarities = cosine_similarity(x, user_vector)
    intent_index = np.argmax(similarities)
    return intents[intent_index]
"""while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    intent = get_intent(user_input)
    print("Predicted Intent:", intent["tag"])
    print("Description:", intent["description"])"""
while True:
    prmt = input("You: ")
    try: print("Bot: The answer is",eval(prmt))
    except:
        # assert prmt.isalpha()
        prmt = prmt.replace(" r ","are").replace(" u ","you").lower().strip("?!.")
        resp = response(prmt)
        print("Bot: " + resp)
