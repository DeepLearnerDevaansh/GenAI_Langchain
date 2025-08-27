from langchain_community.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
# Choose your embedding model
embedding_model_name = "facebook/bart-large-cnn"

# Create the embedder
embedder = HuggingFaceEmbeddings(model_name=embedding_model_name)

# List of documents
documents = [
    "The capital of India is New Delhi.",
    "Machine learning is fun!",
    "The quick brown fox jumps over the lazy dog."
]

# Embed documents
document_embeddings = embedder.embed_documents(documents)

# Query
query = "What is the capital of India?"

# Embed query
query_embedding = embedder.embed_query(query)

scores = cosine_similarity([query_embedding], document_embeddings)
print(scores)