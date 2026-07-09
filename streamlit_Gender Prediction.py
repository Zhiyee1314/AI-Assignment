import streamlit as st
import nltk
from nltk import NaiveBayesClassifier
from nltk.classify import apply_features
from joblib import load, dump
import os
import random

# Download NLTK resources if not already downloaded
nltk.download('names')

# Function to extract features from a name
def extract_gender_features(name):
    name = name.lower()
    features = {
        "suffix": name[-1:],
        "suffix2": name[-2:] if len(name) > 1 else name[0],
        "suffix3": name[-3:] if len(name) > 2 else name[0],
        "suffix4": name[-4:] if len(name) > 3 else name[0],
        "suffix5": name[-5:] if len(name) > 4 else name[0],
        "suffix6": name[-6:] if len(name) > 5 else name[0],
        "prefix": name[:1],
        "prefix2": name[:2] if len(name) > 1 else name[0],
        "prefix3": name[:3] if len(name) > 2 else name[0],
        "prefix4": name[:4] if len(name) > 3 else name[0],
        "prefix5": name[:5] if len(name) > 4 else name[0]
    }
    return features

# Train a new model from scratch
def train_new_model():
    """Train a new Naive Bayes classifier from NLTK names dataset"""
    from nltk import names
    
    st.info("🔄 Training new model... This may take a moment.")
    
    # Get training data from NLTK
    male_names = [(name, 'male') for name in names.words('male.txt')]
    female_names = [(name, 'female') for name in names.words('female.txt')]
    
    # Combine and shuffle
    labeled_names = male_names + female_names
    random.shuffle(labeled_names)
    
    # Extract features
    featuresets = [(extract_gender_features(name), gender) for (name, gender) in labeled_names]
    
    # Train classifier
    classifier = NaiveBayesClassifier.train(featuresets)
    
    # Save model
    dump(classifier, 'gender_prediction.joblib')
    st.success("✅ Model trained and saved successfully!")
    
    return classifier

# Load model with error handling
@st.cache_resource
def load_model():
    model_path = 'gender_prediction.joblib'
    
    # Try to load existing model
    if os.path.exists(model_path):
        try:
            bayes = load(model_path)
            return bayes
        except Exception as e:
            st.warning(f"⚠️ Error loading model: {str(e)}")
            st.info("Retraining model from scratch...")
            # Delete corrupted file
            os.remove(model_path)
            return train_new_model()
    else:
        st.warning("Model file not found.")
        st.info("Training new model from NLTK dataset...")
        return train_new_model()

# Load the trained classifier
bayes = load_model()

# Streamlit app
def main():
    st.title('Gender Prediction App')
    st.write('Enter a name to predict its gender.')
    
    # Input for name
    input_name = st.text_input('Name:')
    
    if st.button('Predict'):
        if input_name.strip() != '':
            # Extract features for the input name
            features = extract_gender_features(input_name)
            
            # Predict using the trained classifier
            predicted_gender = bayes.classify(features)
            
            # Get confidence
            probabilities = bayes.prob_classify(features)
            confidence = probabilities.prob(predicted_gender)
            
            # Display prediction with confidence
            st.success(f'✨ The predicted gender for **"{input_name}"** is: **{predicted_gender.upper()}**')
            st.info(f'Confidence: {confidence:.1%}')
        else:
            st.warning('⚠️ Please enter a name.')

if __name__ == '__main__':
    main()
