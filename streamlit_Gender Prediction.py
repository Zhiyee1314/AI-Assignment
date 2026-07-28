import random
import joblib
import nltk
from nltk import NaiveBayesClassifier
from nltk.corpus import names

nltk.download('names')

# Must match the feature extraction logic in Streamlit
def extract_gender_features(name):
    name = name.lower()
    return {
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

# Load training data
labeled_names = ([(name, 'male') for name in names.words('male.txt')] +
                 [(name, 'female') for name in names.words('female.txt')])

random.seed(42)
random.shuffle(labeled_names)

# Extract features and train NLTK classifier
featuresets = [(extract_gender_features(n), gender) for (n, gender) in labeled_names]
classifier = NaiveBayesClassifier.train(featuresets)

# Save trained classifier
joblib.dump(classifier, 'gender_prediction.joblib')
print("Model saved successfully!")
