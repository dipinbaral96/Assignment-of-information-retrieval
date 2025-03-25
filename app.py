from flask import Flask, render_template, request, jsonify
import joblib
import re
import string
import nltk
import csv
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

app = Flask(__name__)

# Ensure NLTK data is downloaded
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')

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

# Load the models and vectorizer for document classification
try:
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    nb_classifier = joblib.load('nb_classifier.pkl')
    lr_classifier = joblib.load('lr_classifier.pkl')
    models_loaded = True
except:
    models_loaded = False
    print("Warning: Classification models not found. Classification features will be disabled.")

# CLASSIFICATION MODULE ROUTES
@app.route('/classification')
def classification_index():
    return render_template('classification_index.html')

@app.route('/classify', methods=['POST'])
def classify():
    if not models_loaded:
        return jsonify({"error": "Classification models not loaded"}), 500
        
    text = request.form['text']
    classifier_name = request.form.get('classifier', 'nb')
    
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # Transform the text
    text_tfidf = vectorizer.transform([processed_text])
    
    # Select classifier
    classifier = nb_classifier if classifier_name == 'nb' else lr_classifier
    
    # Predict the category
    prediction = classifier.predict(text_tfidf)[0]
    
    # Get probability scores
    probabilities = classifier.predict_proba(text_tfidf)[0]
    prob_dict = {category: float(prob) for category, prob in zip(classifier.classes_, probabilities)}
    
    # Format the result
    result = {
        'prediction': prediction,
        'probabilities': prob_dict,
        'input_text': text
    }
    
    return jsonify(result)

# SEARCH MODULE ROUTES
# Read data from CSV (train_test.csv)
def get_search_results(query):
    results = []
    try:
        with open('files/train_test.csv', mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Basic filtering based on query
                if query.lower() in row['title'].lower():
                    results.append(row)
    except:
        print("Warning: train_test.csv not found or could not be read")
    return results

@app.route('/crawler')
def crawler_index():
    return render_template('crawler_index.html')

@app.route('/search')
def search():
    query = request.args.get('query', '')
    results = get_search_results(query)

    # Sanitize the results before returning
    for result in results:
        # Ensure the abstract is valid and not 'NaN'
        if not result.get('abstract') or result['abstract'] == 'NaN':
            result['abstract'] = 'No abstract available.'

        # Ensure the authors field is a list and not 'NaN'
        if not result.get('authors') or result['authors'] == 'NaN':
            result['authors'] = []

    # Use jsonify instead of json.dumps to automatically set the content-type to application/json
    return jsonify(results)

# MAIN MENU ROUTE
@app.route('/')
def index():
    return render_template('main.html') 

if __name__ == '__main__':
    app.run(debug=True)
