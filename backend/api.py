from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# --- Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_CHECKPOINT = os.path.join(script_dir, "results_trainer_with_test/checkpoint-676")
MAX_LEN = 128

# --- Initialize Flask App ---
app = Flask(__name__)
CORS(app)

# --- Load Model and Tokenizer (Done once on startup) ---
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT)
model.eval()
print("Model loaded successfully.")

# --- Define the Prediction Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    user_input = data.get('text', '')

    if not user_input:
        return jsonify({'error': 'No text provided'}), 400

    # Tokenize and predict
    inputs = tokenizer(
        user_input,
        max_length=MAX_LEN,
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    predicted_class_id = torch.argmax(logits, dim=1).item()
    prediction = "Toxic" if predicted_class_id == 1 else "Non-toxic"
    
    probabilities = torch.softmax(logits, dim=1).squeeze().tolist()
    confidence = probabilities[predicted_class_id]
    
    # Return the result as JSON
    return jsonify({
        'prediction': prediction,
        'confidence': f"{confidence:.4f}"
    })

if __name__ == '__main__':
    app.run(port=5000, debug=True)