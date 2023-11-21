import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
import json
import random
from sklearn.linear_model import LogisticRegression
import pickle
from flask import Flask, request, jsonify

lemmatizer = WordNetLemmatizer()

# load data
with open('intents.json') as file:
    intents = json.loads(file.read())

words = []
classes = []
documents = []
ignore_words = ['?']

# preprocess data
for intent in intents:
    for pattern in intent['patterns']:

        w = nltk.word_tokenize(pattern)
        words.extend(w)

        documents.append((w, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

training = []

# stem and lemmatize
for doc in documents:

    pattern_words = doc[0]
    w = doc[0]
    w = [lemmatizer.lemmatize(word.lower()) for word in w]

    # bag of words
    bag = [0] * len(words)

    for w in words:
        if w in pattern_words:
            bag[words.index(w)] = 1

    output_row = classes.index(doc[1])

    training.append([bag, output_row])

random.shuffle(training)

# split data
train_x = []
train_y = []

for row in training:
    train_x.append(row[0])
    train_y.append(row[1])

train_x = np.array(train_x)
train_y = np.array(train_y)

# train model
model = LogisticRegression().fit(train_x, train_y)

# save model
pickle.dump(model, open('model.pkl', 'wb'))


# classify function
def classify(sentence):
    #  make sure to also resize to 2D array
    bag = [0] * len(words)
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower())
                      for word in sentence_words]

    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1

    bag = np.array(bag).reshape(1, -1)
    result = model.predict(bag)
    result = classes[result[0]]
    return result


# print(classify("what are the symptoms of cold"))
# print(classify("i have fever"))


# generate responses
def response(sentence):
    tag = classify(sentence)
    for intent in intents:
        if intent['tag'] == tag:
            responses = intent['responses']
            return random.choice(responses)


# print(classify('cough, fatigue, panic'))
# print(response("cough, fatigue, panic"))

# receive sentence from user through request and send response and classify as combined json


app = Flask(__name__)


@app.route('/chat', methods=['POST'])
def chat():
    sentence = request.get_json()['sentence']
    classification = classify(sentence)
    responses = response(sentence)

    return json.dumps({
        'response': classification+'. '+responses
    })


if __name__ == '__main__':
    app.run(debug=True)
