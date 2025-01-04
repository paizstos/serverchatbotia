from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Fonction pour charger le modèle depuis Google Drive
def load_model_from_drive(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Le chemin {model_path} n'existe pas")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return tokenizer, model

# Charger le modèle
model_path = 'https://drive.google.com/drive/folders/1gLB1_MwDuGWjJY63dF6-uuILFK2Rnku2?usp=sharing'
tokenizer, model = load_model_from_drive(model_path)

app = Flask(__name__)
CORS(app)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "")
    inputs = tokenizer.encode(user_input, return_tensors="pt")
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run()
