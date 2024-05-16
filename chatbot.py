import pickle
import json
import numpy as np
from tensorflow.keras.models import load_model

# Load tokenizer and label encoder
try:
    with open('tokenizer.pickle', 'rb') as file:
        tokenizer = pickle.load(file)

    with open('label_encoder.pickle', 'rb') as file:
        label_encoder = pickle.load(file)
except Exception as e:
    print("Error loading preprocessing components:", str(e))
    exit()

# Load neural network model
try:
    model = load_model('neural_network_model.h5')
except Exception as e:
    print("Error loading neural network model:", str(e))
    exit()

# Load intents from intents.json
try:
    with open('intents.json', 'r') as file:
        intents_data = json.load(file)
except Exception as e:
    print("Error loading intents data:", str(e))
    exit()

# Function to preprocess user input using tokenizer
def preprocess_input(user_input):
    user_input_sequence = tokenizer.texts_to_sequences([user_input])
    user_input_padded = pad_sequences(user_input_sequence, maxlen=max_length, padding='post')
    return user_input_padded

# Function to predict intent using neural network model
def predict_intent(user_input):
    preprocessed_input = preprocess_input(user_input)
    intent_probabilities = model.predict(preprocessed_input)[0]
    predicted_intent_index = np.argmax(intent_probabilities)
    predicted_intent = label_encoder.classes_[predicted_intent_index]
    return predicted_intent

# Function to retrieve response based on intent
def get_response(predicted_intent):
    for intent in intents_data['intents']:
        if intent['tag'] == predicted_intent:
            responses = intent['responses']
            return np.random.choice(responses)

# Main function to handle user input
def main():
    while True:
        user_input = input("User: ")
        predicted_intent = predict_intent(user_input)
        response = get_response(predicted_intent)
        print("Bot:", response)

if __name__ == "__main__":
    main()
