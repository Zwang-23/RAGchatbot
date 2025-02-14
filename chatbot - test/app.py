# app2.py
import json
import shutil
from flask import Flask, request, jsonify, render_template, session, Response, stream_with_context, copy_current_request_context
import os
from openai import OpenAI
import pypdf
import docx
import create_db
import uuid
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Required for session management

load_dotenv('.env')
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Session cleanup scheduler
scheduler = BackgroundScheduler()
scheduler.start()


def cleanup_old_sessions():
    now = datetime.now()
    sessions_dir = "user_sessions"
    if not os.path.exists(sessions_dir):
        return

    for session_id in os.listdir(sessions_dir):
        session_path = os.path.join(sessions_dir, session_id)
        last_accessed = datetime.fromtimestamp(os.path.getmtime(session_path))
        if (now - last_accessed) > timedelta(minutes=3):
            shutil.rmtree(session_path, ignore_errors=True)


scheduler.add_job(cleanup_old_sessions, 'interval', minutes=1)

@app.before_request
def initialize_session():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
        session_dir = os.path.join("user_sessions", session['session_id'])
        session_data = os.path.join(session_dir, "uploaded-files")
        session_chroma = os.path.join(session_dir, "chroma")

        os.makedirs(session_data, exist_ok=True)
        os.makedirs(session_chroma, exist_ok=True)

        session['DATA_PATH'] = session_data
        session['CHROMA_PATH'] = session_chroma
        session['history'] = []
        session['uploaded_files'] = []
        session['has_documents'] = False


def extract_text_from_file(file_path: str) -> str:
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_path.endswith(('.txt', '.md')):
        return extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {os.path.basename(file_path)}")


def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    with open(file_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text()
    return text


def extract_text_from_docx(file_path: str) -> str:
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])


def extract_text_from_txt(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def format_history(history: list) -> str:
    formatted = []
    for msg in history[1:]:
        role = "User" if msg['role'] == 'user' else "Assistant"
        formatted.append(f"{role}:  {msg['content']}")
    return "\n".join(formatted)

def format_markdown(text: str) -> str:
    """Markdown增强格式化"""
    # 自动检测代码块（需要安装markdown库）
    import markdown
    extensions = ['fenced_code', 'codehilite']
    return markdown.markdown(text, extensions=extensions)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/api/reset', methods=['POST'])
def reset_session():
    # 清理旧会话数据
    old_session_id = session.get('session_id')
    old_session_dir = os.path.join("user_sessions", old_session_id)

    # 生成新会话
    session.clear()
    session['session_id'] = str(uuid.uuid4())
    session_dir = os.path.join("user_sessions", session['session_id'])
    session_data = os.path.join(session_dir, "uploaded-files")
    session_chroma = os.path.join(session_dir, "chroma")

    os.makedirs(session_data, exist_ok=True)
    os.makedirs(session_chroma, exist_ok=True)

    session['DATA_PATH'] = session_data
    session['CHROMA_PATH'] = session_chroma
    session['history'] = []
    session['uploaded_files'] = []
    session['has_documents'] = False

    # 删除旧目录
    if os.path.exists(old_session_dir):
        shutil.rmtree(old_session_dir, ignore_errors=True)

    return jsonify({'status': 'session reset'})


@app.route('/api/upload', methods=['POST'])
def handle_upload():
    if len(session.get('uploaded_files', [])) >= 1:
        return jsonify({'error': 'Only one file allowed per session'}), 400

    DATA_PATH = session.get('DATA_PATH')
    CHROMA_PATH = session.get('CHROMA_PATH')
    uploaded_file = request.files.get('file')

    if uploaded_file:
        file_path = os.path.join(DATA_PATH, uploaded_file.filename)
        uploaded_file.save(file_path)
        try:
            create_db.create_data(DATA_PATH, CHROMA_PATH)
            session['uploaded_files'] = session.get('uploaded_files', []) + [uploaded_file.filename]
            session['has_documents'] = True
            session.modified = True
            return jsonify({
                'response': f"📁 File '{uploaded_file.filename}'  processed successfully!",
                'filename': uploaded_file.filename
            })
        except Exception as e:
            print(f"File processing error: {e}")
            return jsonify({'error': str(e)}), 400
    return jsonify({'error': 'No file provided'}), 400


@app.route('/api/stream')
def stream_response():
    user_message = request.args.get("message")
    CHROMA_PATH = session.get('CHROMA_PATH')
    has_documents = session.get('has_documents', False)

    @copy_current_request_context
    def generate():
        try:
            prompt_sections = []
            rag_context = ""

            if has_documents and os.path.exists(CHROMA_PATH):
                relevant_docs = create_db.query_collection(
                    query_text=user_message,
                    chroma_path=CHROMA_PATH,
                    k=5
                )
                rag_context = "\n".join([doc.page_content for doc in relevant_docs])
                prompt_sections.append(f"DOCUMENT  CONTEXT:\n{rag_context}")

            history = session.get('history', [])
            if len(history) > 1:
                prompt_sections.append(f"CONVERSATION  HISTORY:\n{format_history(history)}")

            prompt_sections.append(f"QUESTION:  {user_message}")
            final_prompt = "\n\n".join(prompt_sections)

            system_message = (
                "You are a helpful assistant specialized in medical research. "
                "If a question is related to medical research, encourage users to upload research papers in pdf file. "
                "If the question is general, please provide a full and complete answer."
            )

            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": final_prompt}
            ]

            full_response = []
            stream = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                stream=True
            )

            for chunk in stream:
                content=chunk.choices[0].delta.content
                if content:
                    yield f"data: {json.dumps({'type': 'stream', 'content': content})}\n\n"
                    full_response.append(content)

            formatted_response = format_markdown(''.join(full_response))
            yield f"data: {json.dumps({'type': 'final', 'content': formatted_response})}\n\n"

            session.setdefault('history', []).extend([
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": ''.join(full_response)}
            ])
            session.modified = True

        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"
            app.logger.error(f"Stream  error: {str(e)}")

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


@app.route('/api/transcribe', methods=['POST'])
def transcribe_audio():
    try:
        audio_file = request.files['audio']
        temp_path = os.path.join(session['DATA_PATH'], "temp_audio.webm")
        audio_file.save(temp_path)

        # Convert to MP3 if needed (requires ffmpeg)
        # os.system(f"ffmpeg  -i {temp_path} {temp_path}.mp3")

        with open(temp_path, "rb") as audio:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio,
                response_format="text"
            )

        return jsonify({
            'text': transcription,
            'status': 'success'
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

if __name__ == '__main__':
    app.run(debug=True)