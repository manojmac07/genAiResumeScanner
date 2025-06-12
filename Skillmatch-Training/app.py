from flask import Flask, render_template, request, jsonify
from genai_backend import process_user_query
from langchain_core.messages import HumanMessage, SystemMessage

app = Flask(__name__)

chat_history = []

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
        handle_frontend_upload(request, storage_client)
        return "Upload processed", 200

@app.route('/get_response', methods=['POST'])
def get_response():
    user_message = request.form['message']
    chat_history.append(HumanMessage(content=user_message))

    # Process the query with the GenAI backend
    bot_response = process_user_query(user_message, chat_history)
    
    chat_history.append(SystemMessage(content=bot_response))

    return jsonify({'response': bot_response})

if __name__ == '__main__':
    app.run(debug=True)