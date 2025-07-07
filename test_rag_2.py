from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS

docs = ['The sky is blue.', 'Grass is green.']
embedding = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
vectors = embedding.embed_documents(docs)
index = FAISS.from_embeddings(vectors, docs)
result = index.similarity_search('What color is the grass?', k=1)
print('RAG result:', result[0].page_content)
