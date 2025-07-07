# ingest_docs.py

from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, CSVLoader, UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader, UnstructuredFileLoader, Docx2txtLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

import os

# 🗂️ Where your documents live
DOCS_PATH = "/data/docs/assistant1"  # Mounted from /mnt/ai-data
DB_PATH = "/data/db/assistant1"

# 🔍 Discover and load supported files
def load_documents(path):
    loaders = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".csv": CSVLoader,
        ".md": UnstructuredMarkdownLoader,
        ".html": UnstructuredHTMLLoader,
        ".docx": Docx2txtLoader,
    }

    documents = []

    for file in os.listdir(path):
        full_path = os.path.join(path, file)
        ext = os.path.splitext(file)[-1].lower()

        loader_cls = loaders.get(ext)
        if loader_cls:
            print(f"📄 Loading: {file}")
            try:
                loader = loader_cls(full_path)
                docs = loader.load()
                documents.extend(docs)
            except Exception as e:
                print(f"⚠️ Failed to load {file}: {e}")
        else:
            print(f"⏭️ Skipping unsupported file type: {file}")
    
    return documents

# 🔪 Chunk the text
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    return splitter.split_documents(documents)

# 🤖 Embed and save to FAISS
def build_vectorstore(chunks):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(DB_PATH)
    print(f"✅ Saved vectorstore to: {DB_PATH}")

# 🚀 Full process
if __name__ == "__main__":
    docs = load_documents(DOCS_PATH)
    if not docs:
        print("❌ No documents loaded. Exiting.")
    else:
        chunks = split_documents(docs)
        build_vectorstore(chunks)
