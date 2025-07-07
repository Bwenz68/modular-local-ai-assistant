# query_docs.py

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Set path to where FAISS index is saved
DB_PATH = "/data/db/assistant1"

# Load FAISS index
embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.load_local(DB_PATH, embeddings=embedding, allow_dangerous_deserialization=True)

# Ask your question
query = input("\n❓ What do you want to know? ")
results = db.similarity_search(query, k=3)

# Show top results
print("\n🔍 Top matching chunks:\n")
for i, result in enumerate(results):
    print(f"Result {i+1}:")
    print(result.page_content)
    print("-" * 40)
