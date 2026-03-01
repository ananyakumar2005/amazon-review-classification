# debug_model.py
import joblib
import numpy as np

# Load the model and components
model = joblib.load('best_sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

print("=== MODEL INFO ===")
print(f"Model type: {type(model).__name__}")
print(f"Classes: {label_encoder.classes_}")

# Test with some text
test_texts = [
    "amazing perfect sound quality great value",  # Should be Positive
    "okay nothing special",  # Should be Neutral  
    "terrible broke waste money",  # Should be Negative
    "good works expected",  # Should be Positive
    "not good disappointed"  # Should be Negative
]

print("\n=== PREDICTION TEST ===")
for text in test_texts:
    # Vectorize the text
    text_vectorized = vectorizer.transform([text])
    
    # Check if any features are found
    print(f"\nText: '{text}'")
    print(f"Features found: {text_vectorized.nnz}")
    
    if text_vectorized.nnz > 0:
        # Make prediction
        prediction_encoded = model.predict(text_vectorized)
        prediction = label_encoder.inverse_transform(prediction_encoded)[0]
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(text_vectorized)[0]
            print(f"Prediction: {prediction}")
            print(f"Probabilities: {dict(zip(label_encoder.classes_, probabilities))}")
        else:
            print(f"Prediction: {prediction}")
    else:
        print("❌ NO FEATURES FOUND - This is the problem!")
        
        # Let's see what features the vectorizer expects
        feature_names = vectorizer.get_feature_names_out()
        print(f"Vectorizer has {len(feature_names)} features")
        print(f"First 20 features: {feature_names[:20]}")