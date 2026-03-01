from flask import Flask, render_template, request, jsonify
import os
import joblib

app = Flask(__name__)

print("🔍 Debug information:")
print(f"Current directory: {os.getcwd()}")
print(f"Files in current directory: {os.listdir('.')}")

# Check if templates folder exists
templates_path = os.path.join(os.getcwd(), 'templates')
print(f"Templates folder exists: {os.path.exists(templates_path)}")
if os.path.exists(templates_path):
    print(f"Files in templates folder: {os.listdir(templates_path)}")

# Check if model files exist
model_files = ['best_sentiment_model.pkl', 'tfidf_vectorizer.pkl', 'label_encoder.pkl']
for file in model_files:
    exists = os.path.exists(file)
    print(f"{file} exists: {exists}")

@app.route('/')
def home():
    print("📄 Home route accessed")
    try:
        return render_template('index.html')
    except Exception as e:
        print(f"❌ Error rendering template: {e}")
        return f"Error: {e}"

@app.route('/test')
def test():
    return "Flask is working! If you see this, the server is running."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        text = request.form.get('review_text', '')
        print(f"📨 Received prediction request: {text[:50]}...")
        
        # Mock response for testing
        return jsonify({
            'success': True,
            'prediction': 'Positive',
            'confidence': '0.850',
            'probabilities': {'Positive': 0.85, 'Neutral': 0.10, 'Negative': 0.05},
            'sentiment_color': 'success'
        })
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    print("\n🚀 Starting DEBUG Flask App...")
    print("📍 Test these URLs:")
    print("   http://localhost:5000/")
    print("   http://localhost:5000/test")
    app.run(debug=True, host='0.0.0.0', port=5000)