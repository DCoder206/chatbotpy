from nltk.metrics import edit_distance
from nltk import word_tokenize
from functools import lru_cache
from random import choice as ranmsg
from json import load as json_load
with open("cybot_dataset.json","r") as file: dataset = json_load(file)
dataset = dataset["intents"]
def response(prompt:str) -> str:
    global dataset
    for ele in dataset:
        if prompt.capitalize() in ele["patterns"]:
            return ranmsg(ele["patterns"])
while True:
    prmt = input("You: ").lower()
    assert prmt.isalpha()
    prmt = prmt.replace(" r ","are").replace(" u ","you")
    if prmt[-1] == " ": prmt = prmt[:-1]
    resp = response(prmt)
    print("Bot: " + resp)
