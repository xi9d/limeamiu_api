from flask import Flask, request, jsonify
import json
import random
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

app = Flask(__name__)

with open('intents.json') as f:
    intents = json.load(f)


def clean_sentence(sentence):
    sentence = sentence.lower()
    sentence = lemmatizer.lemmatize(sentence)
    return sentence


def intent_classifier(sentence):
    sentence = clean_sentence(sentence)

    for intent in intents:
        for pattern in intent['patterns']:
            pattern = clean_sentence(pattern)
            if pattern in sentence:
                return intent

    return None


def bot_response(sentence):
    intent = intent_classifier(sentence)

    if intent:
        return random.choice(intent['responses'])
    else:
        return "I did not understand"


@app.route('/chatbot', methods=['POST'])
def chatbot():
    prompt = request.json['prompt']
    prompt = clean_sentence(prompt)
    intent = intent_classifier(prompt)
    response = bot_response(prompt)

    return jsonify({'response': response})


if __name__ == '__main__':
    app.run(debug=True)