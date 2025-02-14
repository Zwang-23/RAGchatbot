from flask import Flask, request, jsonify, render_template, session
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Add this crucial line

# Set your OpenAI API key
from openai import OpenAI

client = OpenAI(api_key="sk-2e026220f64e457699bc465cf49c9e91", base_url="https://api.deepseek.com")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api', methods=['POST'])
def chat_api():
    user_message = request.form.get("message")
    uploaded_file = request.files.get('file')

    if uploaded_file:
        os.makedirs('uploaded-files', exist_ok=True)
        file_path = os.path.join('uploaded-files', uploaded_file.filename)
        uploaded_file.save(file_path)
        return jsonify({'response': f"File '{uploaded_file.filename}' uploaded successfully!"})

    if 'conversation_history' not in session:
        session['conversation_history'] = [
            {"role": "system", "content": "You are an experienced and knowledgeable medical researcher."}
        ]

    session['conversation_history'].append({"role": "user", "content": user_message})
    session['conversation_history'] = session['conversation_history'][-10:]

    # Use OpenAI's API to get a response
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=session['conversation_history']
    )

    reply = response.choices[0].message.content
    session['conversation_history'].append({"role": "assistant", "content": reply})
    session.modified = True
    return jsonify({'response': reply})


if __name__ == '__main__':
    app.run(debug=True)