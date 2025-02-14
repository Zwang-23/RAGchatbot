from flask import Flask, request, jsonify, render_template
import os
import numpy as np
from openai import OpenAI
import PyPDF2
import docx
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
client = OpenAI(api_key="sk-2e026220f64e457699bc465cf49c9e91", base_url="https://api.deepseek.com")

# Global context with vector store
conversation_context = {
    'history': [
        {"role": "system", "content": "You are a helpful medical research assistant."}
    ],
    'vector_store': [],
    'chunk_size': 1000  # Optimal chunk size for OpenAI embeddings
}


def chunk_text(text, chunk_size=1000, overlap=100):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        start = end - overlap
    return chunks


def get_embedding(text):
    """Get OpenAI embedding for a text chunk"""
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )
    return response.data[0].embedding


def process_document(file_path):
    """Process document and create vector embeddings"""
    raw_text = extract_text_from_file(file_path)
    chunks = chunk_text(raw_text, chunk_size=conversation_context['chunk_size'])

    # Generate and store embeddings
    for chunk in chunks:
        embedding = get_embedding(chunk)
        conversation_context['vector_store'].append({
            'text': chunk,
            'embedding': embedding
        })


def find_relevant_context(query, top_k=3):
    """Retrieve top-k most relevant chunks using OpenAI embeddings"""
    query_embedding = get_embedding(query)
    similarities = []

    for doc in conversation_context['vector_store']:
        sim = cosine_similarity([query_embedding], [doc['embedding']])[0][0]
        similarities.append((sim, doc['text']))

    # Return top-k most relevant chunks
    similarities.sort(reverse=True, key=lambda x: x[0])
    return [text for _, text in similarities[:top_k]]


@app.route('/api', methods=['POST'])
def chat_api():
    user_message = request.form.get("message")
    uploaded_file = request.files.get('file')

    if uploaded_file:
        # Save and process document
        os.makedirs('uploads', exist_ok=True)
        file_path = os.path.join('uploads', uploaded_file.filename)
        uploaded_file.save(file_path)
        process_document(file_path)
        return jsonify({'response': f"File '{uploaded_file.filename}' processed successfully!"})

    # Retrieve relevant context using RAG
    relevant_chunks = find_relevant_context(user_message)

    # Build the augmented prompt
    messages = [conversation_context['history'][0]]

    # Add relevant document context
    if relevant_chunks:
        context = "\n\nDocument references:\n" + "\n\n".join([
            f"[Excerpt {i + 1}]: {chunk}" for i, chunk in enumerate(relevant_chunks)
        ])
        messages.append({"role": "system", "content": context})

    # Add conversation history (last 4 messages)
    messages.extend(conversation_context['history'][1:][-9:])
    messages.append({"role": "user", "content": user_message})

    # Generate response using ChatGPT
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=0.7
    )

    reply = response.choices[0].message.content

    # Update conversation history
    conversation_context['history'].extend([
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": reply}
    ])

    return jsonify({'response': reply})


# Keep existing extract_text_from_file function from previous code

if __name__ == '__main__':
    app.run(debug=True)