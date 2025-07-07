# test_rag.py

# Step 1: Import everything we need
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

# Step 2: Create some documents (later we'll load real PDFs)
docs = ["The sky is blue.", "Grass is green.", "The sun is bright."]

# Step 3: Split each sentence into manageable pieces
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = splitter.create_documents(docs)

# Step 4: Use a model to turn each sentence into a set of numbers (embeddings)
embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Step 5: Create a FAISS database from the embeddings
db = FAISS.from_documents(split_docs, embedding)

# Step 6: Ask a question
query = "What color is the grass?"
results = db.similarity_search(query, k=1)

# Step 7: Show the answer
print("\n✅ Query:", query)
print("🔍 Top Result:", results[0].page_content)
