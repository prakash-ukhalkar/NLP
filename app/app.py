import joblib
from flask import Flask, request, jsonify
import os

# Define paths relative to the app.py location
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'svm_text_classifier.pkl')
NAMES_PATH = os.path.join(MODEL_DIR, 'target_names.pkl')

app = Flask(__name__)

# --- Load Model on Startup ---
try:
    model = joblib.load(MODEL_PATH)
    target_names = joblib.load(NAMES_PATH)
    print("Model and Target Names loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# --- Define API Endpoints ---

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request (expected JSON with 'text' key)
    data = request.get_json(force=True)
    input_text = data.get('text', '')

    if not model:
        return jsonify({'error': 'Model not loaded'}), 500
        
    if not input_text:
        return jsonify({'error': 'No text provided'}), 400

    try:
        # The pipeline handles tokenization, vectorization, and classification
        prediction_id = model.predict([input_text])[0]
        predicted_label = target_names[prediction_id]
        
        return jsonify({
            'status': 'success',
            'input': input_text,
            'prediction': predicted_label
        })
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {e}'}), 500

@app.route('/')
def home():
    return "NLP Classification Service is running! Use POST /predict."

if __name__ == '__main__':
    # For production, use a WSGI server like Gunicorn. For local testing:
    app.run(debug=True, host='0.0.0.0', port=5000)