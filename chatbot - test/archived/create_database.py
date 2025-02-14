# create_database.py
import os
import shutil
import time
import errno
import threading
from typing import List
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import uuid




# Add a lock for Chroma operations
chroma_lock = threading.Lock()


def safe_delete_chroma(chroma_path: str):
    """Delete Chroma directory with retries for Windows file locking issues."""
    if not os.path.exists(chroma_path):
        return

    for _ in range(5):  # Retry up to 5 times
        try:
            shutil.rmtree(chroma_path)
            return
        except OSError as e:
            if e.errno in (errno.EBUSY, errno.EACCES, errno.EPERM):
                print(f"Waiting for Chroma files to unlock... (attempt {_ + 1}/5)")
                time.sleep(0.5)
            else:
                raise


def save_to_chroma(chunks: List[Document], chroma_path: str):
    with chroma_lock:
        safe_delete_chroma(chroma_path)  # Use passed path
        Chroma.from_documents(
            chunks,
            OpenAIEmbeddings(),
            persist_directory=chroma_path,
            collection_metadata = {"metadata_fields": "id"}
        )


def query_collection(query_text: str, chroma_path: str, k: int = 5) -> List[Document]:
    with chroma_lock:  # Acquire lock before querying
        if not os.path.exists(chroma_path):
            return []

        db = Chroma(
            persist_directory=chroma_path,
            embedding_function=OpenAIEmbeddings()
        )
        return db.similarity_search(query_text, k=k)


def create_data(data_path: str, chroma_path: str):
    generate_data_store(data_path, chroma_path)

def generate_data_store(data_path: str, chroma_path: str):
    documents = load_documents(data_path)
    chunks = split_text(documents)
    save_to_chroma(chunks, chroma_path)


def load_documents(data_path: str) -> List[Document]:
    loaders = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".md": TextLoader,
        ".docx": UnstructuredFileLoader
    }

    documents = []
    for file in os.listdir(data_path):
        file_path = os.path.join(data_path, file)
        ext = os.path.splitext(file)[1].lower()
        if ext in loaders:
            loader = loaders[ext](file_path)
            docs = loader.load()

            for doc in docs:
                unique_id = str(uuid.uuid4())
                doc.metadata["id"] = unique_id
                setattr(doc, "id", unique_id)

            documents.extend(docs)

    return documents


def split_text(documents: list[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True,
    )
    split_docs = text_splitter.split_documents(documents)
    for split_doc in split_docs:
        # Ensure each split document has an id in metadata and as an attribute.
        if "id" not in split_doc.metadata:
            unique_id = str(uuid.uuid4())
            split_doc.metadata["id"] = unique_id
            setattr(split_doc, "id", unique_id)
    return split_docs



