import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- Configuration ---
MODEL_CHECKPOINT = "./results_trainer_with_test/checkpoint-676"
MAX_LEN = 128

# --- Load Model and Tokenizer ---
print("Loading model and tokenizer...")
try:
    # Load the tokenizer from the checkpoint
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    # Load the model from the checkpoint
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT)
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    exit()

# Set the model to evaluation mode
model.eval()

print("Model loaded successfully. You can now enter messages for classification.")
print("Type 'quit' to exit.")

# --- Interactive Loop ---
while True:
    # Get user input from the command line
    user_input = input("\nEnter a message to classify: ")

    if user_input.lower() == 'quit':
        break

    # Tokenize the input text
    inputs = tokenizer(
        user_input,
        max_length=MAX_LEN,
        truncation=True,
        return_tensors="pt"  # Return PyTorch tensors
    )

    # Make a prediction
    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model(**inputs)
        logits = outputs.logits

    # Get the predicted class
    predicted_class_id = torch.argmax(logits, dim=1).item()
    prediction = "Toxic" if predicted_class_id == 1 else "Non-toxic"

    # Get the probabilities
    probabilities = torch.softmax(logits, dim=1).squeeze().tolist()
    confidence = probabilities[predicted_class_id]

    print(f"\nClassification: {prediction}")
    print(f"Confidence: {confidence:.4f}")
    print("-" * 20)

print("Exiting demo.")