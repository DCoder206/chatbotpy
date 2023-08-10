import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from json import load as json_load
from random import choice
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")
with open("cybot_dataset.json","r") as file: dataset = json_load(file)

def response(prompt:str) -> str:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(dataset['intents']))

    inputs = tokenizer(prompt, return_tensors="tf", padding=True, truncation=True)

    logits = model(inputs['input_ids'], training=False).logits
    predicted_class = tf.argmax(logits, axis=1).numpy()[0]

    predicted_tag = dataset['intents'][predicted_class]['tag']
    for intent in dataset['intents']:
        if intent['tag'] == predicted_tag:
            responses = intent['responses']
            bot_response = choice(responses)
            return bot_response
while True:
    prmt = input("You: ").lower()
    assert prmt.isalpha()
    prmt = prmt.replace(" r ","are").replace(" u ","you")
    if prmt[-1] == " ": prmt = prmt[:-1]
    resp = response(prmt)
    print("Bot: " + resp)
