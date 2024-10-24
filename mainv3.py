from functools import lru_cache
from json import load as json_load
from re import sub

import gensim.downloader as api
from nltk import edit_distance
from numpy import argmax, mean
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from spacy import load

with open("cybot.json","r") as file: dataset = json_load(file)
model = api.load("word2vec-google-news-300")
def literal_match(prompt:str):
    global dataset
    matched_diseases = {}
    for ele in dataset["diseases"]:
        if prompt.capitalize() in ele["symptoms"]:
            matched_diseases[ele["name"]] = ele["treatments"]
    return matched_diseases if matched_diseases else None
def word_embeddings(prompt:str):
    patterns_corpus = [" ".join(intent["symptoms"]) for intent in dataset["diseases"]]
    pattern_embeddings = []
    for pattern in patterns_corpus:
        words = pattern.lower().split()
        embeddings = [model[word] for word in words if word in model]
        if embeddings:
            avg_embedding = mean(embeddings, axis=0)
            pattern_embeddings.append(avg_embedding)
        else:
            pattern_embeddings.append(None)
    user_words = prompt.lower().split()
    user_embedding = mean([model[word] for word in user_words if word in model], axis=0)
    similarities = cosine_similarity(pattern_embeddings, [user_embedding])
    closest_intent_index = argmax(similarities)
    closest_similarity = similarities[closest_intent_index][0]
    threshold = 0.25
    matched_diseases = {}
    if closest_similarity > threshold:
        disease_name = dataset["diseases"][closest_intent_index]["name"]
        treatments = dataset["diseases"][closest_intent_index]["treatments"]
        matched_diseases[disease_name] = treatments
    return matched_diseases if matched_diseases else None
def nearest_word(prompt:str):
    user_words = prompt.lower().split()
    user_embedding = mean([model[word] for word in user_words if word in model], axis=0)
    patterns_corpus = [" ".join(intent["symptoms"]) for intent in dataset["diseases"]]
    pattern_embeddings = []
    for pattern in patterns_corpus:
        words = pattern.lower().split()
        embeddings = [model[word] for word in words if word in model]
        if embeddings:
            avg_embedding = mean(embeddings, axis=0)
            pattern_embeddings.append(avg_embedding)
        else:
            pattern_embeddings.append(None)
    num_neighbors = 3
    nbrs = NearestNeighbors(n_neighbors=num_neighbors, metric="cosine").fit(pattern_embeddings)
    distances, indices = nbrs.kneighbors([user_embedding])
    threshold = 0.25
    matched_diseases = {}
    for i in range(num_neighbors):
        similarity = 1 - distances[0][i]
        if similarity > threshold:
            intent_index = indices[0][i]
            disease_name = dataset["diseases"][intent_index]["name"]
            treatments = dataset["diseases"][intent_index]["treatments"]
            matched_diseases[disease_name] = treatments
    return matched_diseases if matched_diseases else None
def vector_match(prompt:str):
    global dataset
    diseases = dataset["diseases"]
    symptoms_corpus = [" ".join(intent["symptoms"]) for intent in diseases]
    tfidf_vectorizer = TfidfVectorizer()
    x_tfidf = tfidf_vectorizer.fit_transform(symptoms_corpus)
    nlp = load("en_core_web_md") # sm # python -m spacy download en_core_web_md
    user_doc = nlp(prompt)
    user_vector = tfidf_vectorizer.transform([" ".join([token.text for token in user_doc])])
    similarities = cosine_similarity(x_tfidf,user_vector)
    intent_index = argmax(similarities)
    max_similarity = similarities[intent_index]
    matched_diseases = {}
    if max_similarity > 0.25:
        disease_name = diseases[intent_index]["name"]
        treatments = diseases[intent_index]["treatments"]
        matched_diseases[disease_name] = treatments
    return matched_diseases if matched_diseases else None
def char_overlap(prompt:str):
    global dataset
    @lru_cache(maxsize=32)
    def edit_dist(s1:str,s2:str,transpos:bool) -> int: return edit_distance(s1,s2,transpositions=transpos)
    max_overlap = 0
    matched_key = None
    for intent in dataset["diseases"]:
        for pattern in intent["symptoms"]:
            if edit_dist(pattern, prompt, True) > 5:
                continue
            overlap = len(set(pattern) & set(prompt))
            if overlap > max_overlap:
                max_overlap = overlap
                matched_key = intent["name"]
    if matched_key:
        matched_intent = None
        for intent in dataset["diseases"]:
            if intent["name"] == matched_key:
                matched_intent = intent
                break
        if matched_intent: matched_diseases = {matched_key: matched_intent["treatments"]}
    return matched_diseases if matched_diseases else None
def disease_search(symp:str) -> list:
    treats = []
    for disease in dataset["diseases"]:
        if symp.capitalize() in disease["symptoms"]:
            treats.append(disease["name"])
    return treats
@lru_cache(maxsize=64)
def response(prompt:str):
    funcs = (literal_match,nearest_word,vector_match,word_embeddings,char_overlap)
    for func in funcs:
        result = func(prompt)
        if result is not None:
            return result
    return "I didn't understand your question"
while True:
    prmt = input("You: ").lower().replace("I have","")
    prmt = sub(r"[^a-zA-Z0-9,\s]", "", prmt)
    resp = response(prmt)
    if "dict" not in str(type(resp)): print(resp)
    else:
        print("Possible diseases and treatments are:")
        for k in resp:
            resp[k] = sub(r"[\[\]']","",str(resp[k]))
            print(f"Disease: {k}\nTreatments: {resp[k]}")
            print()
    # respstr = ""
    # for i,v in enumerate(disease_search(prmt)):
    #    respstr += f"{str(i + 1)}) {v}\n"
    # print(f"Bot: Possible diseases are:\n{respstr}")
