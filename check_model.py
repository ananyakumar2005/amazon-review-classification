import joblib
import os

files = ['best_sentiment_model.pkl', 'tfidf_vectorizer.pkl', 'label_encoder.pkl']

for file in files:
    exists = os.path.exists(file)
    print(f"{file}: {'✅ EXISTS' if exists else '❌ MISSING'}")
    
    if exists:
        try:
            obj = joblib.load(file)
            print(f"  Type: {type(obj).__name__}")
            if hasattr(obj, 'classes_'):
                print(f"  Classes: {obj.classes_}")
        except Exception as e:
            print(f"  ❌ Error loading: {e}")