from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

@app.route("/")
def index():
    return render_template('chat.html')  # Serve the chat page

@app.route("/get", methods=["POST"])
def chat():
    # Get the message from the form data
    user_message = request.form.get('msg', '')
    if not user_message.strip():
        return jsonify({'response': "Please type something."})
    response = get_chat_response(user_message)
    return jsonify({'response': response})

def get_chat_response(text):
    # Encode input and generate reply in one call
    new_user_input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors='pt')
    chat_history_ids = model.generate(
        new_user_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id
    )
    # Decode the response (skip the input part)
    reply = tokenizer.decode(
        chat_history_ids[:, new_user_input_ids.shape[-1]:][0],
        skip_special_tokens=True
    )
    return reply

if __name__ == '__main__':
    # Change host to '0.0.0.0' so it's accessible from other devices on the network
    app.run(debug=True, host='0.0.0.0')
