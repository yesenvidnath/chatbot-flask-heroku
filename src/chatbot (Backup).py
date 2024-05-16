import pickle
import nltk
import string
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.models import load_model

# Load NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load the trained decision tree model
def load_decision_tree_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Load the neural network model
def load_neural_network_model(model_path):
    model = load_model(model_path)
    return model

# Preprocess user input using NLP techniques
def preprocess_input(user_input):
    # Tokenize input
    tokens = nltk.word_tokenize(user_input.lower())

    # Remove punctuation
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]

    # Remove stopwords
    filtered_words = [word for word in stripped if word not in stop_words]

    # Join words back into a sentence
    preprocessed_input = ' '.join(filtered_words)

    return preprocessed_input

# Use the neural network model to predict the PC part based on user input
def predict_pc_part_neural_network(user_input, model, vectorizer):
    preprocessed_input = preprocess_input(user_input)
    vectorized_input = vectorizer.transform([preprocessed_input])
    predicted_part_index = np.argmax(model.predict(vectorized_input))
    predicted_part = vectorizer.get_feature_names_out()[predicted_part_index]
    return predicted_part

# Main function to run the chatbot
def main():
    # Load the trained decision tree model
    decision_tree_model_path = 'models/decision_tree_model.pkl'
    decision_tree_model = load_decision_tree_model(decision_tree_model_path)

    # Load the neural network model
    neural_network_model_path = 'models/neural_network_model.h5'
    neural_network_model = load_neural_network_model(neural_network_model_path)

    # Load the CountVectorizer
    with open('models/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    # Accept user input
    while True:
        user_input = input("User: ")
        
        # Check for exit command
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break

        # Predict PC part based on user input using neural network
        predicted_part_nn = predict_pc_part_neural_network(user_input, neural_network_model, vectorizer)

        # Predict PC part based on user input using decision tree
        predicted_part_dt = predict_pc_part_decision_tree(user_input, decision_tree_model)

        # Return the predicted PC part to the user
        print("Chatbot (Neural Network): Predicted PC Part:", predicted_part_nn)
        print("Chatbot (Decision Tree): Predicted PC Part:", predicted_part_dt)

if __name__ == "__main__":
    main()
