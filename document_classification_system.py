import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Load the data
df = pd.read_csv('./files/bbc.csv')


# Clean the data
df = df.dropna()  # Remove any rows with NaN values

# Text preprocessing function
def preprocess_text(text):
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = re.sub(f'[{string.punctuation}]', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Tokenize
        tokens = nltk.word_tokenize(text)
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        # Stemming
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]
        return ' '.join(tokens)
    return ''

# Apply preprocessing
df['processed_text'] = df['Text'].apply(preprocess_text)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    df['processed_text'], 
    df['Category'], 
    test_size=0.2, 
    random_state=42,
    stratify=df['Category']  # Ensure proportional split across categories
)

# Feature extraction
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train classifiers
# 1. Naive Bayes
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)

# 2. Logistic Regression
lr_classifier = LogisticRegression(max_iter=1000)
lr_classifier.fit(X_train_tfidf, y_train)

# Evaluate models
nb_predictions = nb_classifier.predict(X_test_tfidf)
lr_predictions = lr_classifier.predict(X_test_tfidf)

print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_predictions))
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_predictions))

print("\nNaive Bayes Classification Report:")
print(classification_report(y_test, nb_predictions))

print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, lr_predictions))

# Save the models and vectorizer
joblib.dump(nb_classifier, 'nb_classifier.pkl')
joblib.dump(lr_classifier, 'lr_classifier.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# Function to classify new text
def classify_text(text, classifier_name='nb'):
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # Load the vectorizer and classifier
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    
    if classifier_name == 'nb':
        classifier = joblib.load('nb_classifier.pkl')
    else:
        classifier = joblib.load('lr_classifier.pkl')
    
    # Transform the text
    text_tfidf = vectorizer.transform([processed_text])
    
    # Predict the category
    prediction = classifier.predict(text_tfidf)[0]
    
    # Get probability scores
    probabilities = classifier.predict_proba(text_tfidf)[0]
    prob_dict = {category: prob for category, prob in zip(classifier.classes_, probabilities)}
    
    return prediction, prob_dict

# Example usage
if __name__ == "__main__":
    sample_text = "Elections are important for democracy and voting is a fundamental right."
    prediction, probabilities = classify_text(sample_text, 'nb')
    print(f"The predicted category is: {prediction}")
    print("Probabilities:")
    for category, prob in probabilities.items():
        print(f"{category}: {prob:.4f}")
