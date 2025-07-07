# query_docs.py

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

import os

# 🗂️ Where your vector DB lives
DB_PATH = "/data/db/assistant1"  # Mounted from /mnt/ai-data

# 🤖 Load embeddings model
embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# 🔍 Function to query the vectorstore
def query_vectorstore(user_query: str):
    try:
        # Load the vectorstore
        db = FAISS.load_local(DB_PATH, embeddings=embedding, allow_dangerous_deserialization=True)
        print(f"Loaded vectorstore from: {DB_PATH}")

        # Perform a similarity search
        docs = db.similarity_search(user_query)

        # Extract page content
        context_chunks = [doc.page_content for doc in docs]
        print(f"Found {len(context_chunks)} relevant chunks.")
        return context_chunks
    except Exception as e:
        print(f"❌ Error querying vectorstore: {e}")
        return []

# Optional: For standalone testing of query_docs.py if you ever need it
if __name__ == "__main__":
    print("Running query_docs.py in standalone test mode.")
    if not os.path.exists(DB_PATH):
        print(f"Vectorstore not found at {DB_PATH}. Please run ingest_docs.py first.")
    else:
        test_query = input("\n❓ What do you want to know? (Standalone test) ")
        results = query_vectorstore(test_query)
        for i, chunk in enumerate(results):
            print(f"\n--- Chunk {i+1} ---")
            print(chunk)
