from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

# Download NLTK data (should work now)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

app = Flask(__name__)

class SentimentPredictor:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.load_model()
    
    def load_model(self):
        """Load the trained model and components"""
        try:
            print("📂 Loading model files...")
            self.model = joblib.load('best_sentiment_model.pkl')
            self.vectorizer = joblib.load('tfidf_vectorizer.pkl')
            self.label_encoder = joblib.load('label_encoder.pkl')
            print("✅ Model loaded successfully!")
            print(f"📊 Model type: {type(self.model).__name__}")
            print(f"🎯 Classes: {self.label_encoder.classes_}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

    def preprocess_text(self, text):
        """
        EXACT same preprocessing as your Jupyter notebook
        """
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Tokenize using NLTK
        tokens = nltk.word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        
        # Remove short words
        tokens = [token for token in tokens if len(token) > 2]
        
        return ' '.join(tokens)

    def predict_sentiment(self, text):
        """Predict sentiment for the given text"""
        if self.model is None:
            return "Error: Model not loaded", 0.0, {}
        
        try:
            print(f"📝 Original text: {text}")
            
            # Preprocess the text using EXACT same method as notebook
            processed_text = self.preprocess_text(text)
            print(f"🔧 Processed text: {processed_text}")
            
            if not processed_text.strip():
                print("⚠️ Empty text after preprocessing")
                return "Neutral", 0.5, {"Positive": 0.33, "Neutral": 0.34, "Negative": 0.33}
            
            # Vectorize the text
            text_vectorized = self.vectorizer.transform([processed_text])
            print(f"📊 Features found: {text_vectorized.nnz}")
            
            # Predict
            prediction_encoded = self.model.predict(text_vectorized)
            prediction = self.label_encoder.inverse_transform(prediction_encoded)[0]
            print(f"🎯 Prediction: {prediction}")
            
            # Get probabilities
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(text_vectorized)[0]
                confidence = max(probabilities)
                prob_dict = dict(zip(self.label_encoder.classes_, probabilities))
                print(f"📈 Probabilities: {prob_dict}")
            else:
                confidence = 0.8
                prob_dict = {cls: 0.0 for cls in self.label_encoder.classes_}
                prob_dict[prediction] = 1.0
            
            return prediction, confidence, prob_dict
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return f"Error during prediction: {str(e)}", 0.0, {}

# Initialize the predictor
predictor = SentimentPredictor()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for sentiment prediction"""
    try:
        text = request.form.get('review_text', '')
        
        if not text.strip():
            return jsonify({'success': False, 'error': 'Please enter a review text'})
        
        print(f"🔍 Received review: {text}")
        
        # Use your trained model for prediction
        prediction, confidence, probabilities = predictor.predict_sentiment(text)
        
        if prediction.startswith('Error'):
            return jsonify({'success': False, 'error': prediction})
        
        # Get sentiment color for UI
        sentiment_colors = {
            'Positive': 'success',
            'Neutral': 'warning', 
            'Negative': 'danger'
        }
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'confidence': f'{confidence:.3f}',
            'probabilities': probabilities,
            'sentiment_color': sentiment_colors.get(prediction, 'secondary')
        })
        
    except Exception as e:
        print(f"❌ Server error: {e}")
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'})

@app.route('/test')
def test_predictions():
    """Test page to verify predictions are working"""
    test_reviews = [
        "These strings are amazing! Perfect sound quality and great value for money. Highly recommended!",
        "The product is okay but nothing special. Does the job but feels a bit cheap.",
        "Terrible quality! Broke after just one week. Complete waste of money. Do not buy!",
        "Good product overall, works as expected. Delivery was fast and packaging was good.",
        "Very disappointed with this purchase. The sound quality is poor and it feels flimsy."
    ]
    
    results_html = "<h1>Test Predictions</h1>"
    results_html += "<p>Testing with sample reviews to verify the model is working correctly:</p>"
    
    for review in test_reviews:
        prediction, confidence, probabilities = predictor.predict_sentiment(review)
        
        color = "green" if prediction == "Positive" else "orange" if prediction == "Neutral" else "red"
        
        results_html += f"""
        <div style="border: 1px solid #ccc; margin: 10px; padding: 10px; border-left: 5px solid {color};">
            <strong>Review:</strong> {review}<br>
            <strong>Prediction:</strong> <span style="color: {color}; font-weight: bold;">{prediction}</span><br>
            <strong>Confidence:</strong> {confidence}<br>
            <strong>Probabilities:</strong> {probabilities}
        </div>
        """
    
    results_html += '<br><a href="/">← Back to Main App</a>'
    return results_html

if __name__ == '__main__':
    print("🚀 Starting Sentiment Analysis Flask App")
    print("📍 Main App: http://localhost:5000")
    print("🧪 Test Page: http://localhost:5000/test")
    print("✅ Using EXACT same preprocessing as your notebook")
    app.run(debug=True, host='0.0.0.0', port=5000)