import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from json import load as json_load

# Define the dataset
with open("cybot_dataset.json","r") as file: dataset = json_load(file)

# Extract patterns and tags from the dataset
patterns = []
tags = []
for intent in dataset["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        tags.append(intent["tag"])

# Initialize CountVectorizer and LabelEncoder
vectorizer = CountVectorizer()
encoder = LabelEncoder()

# Fit and transform patterns and tags
X = vectorizer.fit_transform(patterns).toarray()
y = encoder.fit_transform(tags)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# Define a simple neural network model
class ChatbotModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(ChatbotModel, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        
    def forward(self, x):
        x = self.fc(x)
        return x

input_size = X_train.shape[1]
output_size = len(np.unique(y_train))
model = ChatbotModel(input_size, output_size)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Inference function
def get_response(user_input):
    user_input = vectorizer.transform([user_input]).toarray()
    user_input = torch.tensor(user_input, dtype=torch.float32)
    with torch.no_grad():
        output = model(user_input)
        predicted_class = torch.argmax(output).item()
    predicted_tag = encoder.classes_[predicted_class]
    
    for intent in dataset["intents"]:
        if intent["tag"] == predicted_tag:
            return random.choice(intent["responses"])
    
    return "I'm not sure how to respond to that."

# Chat loop
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Bot: Goodbye!")
        break
    response = get_response(user_input)
    print("Bot:", response)
