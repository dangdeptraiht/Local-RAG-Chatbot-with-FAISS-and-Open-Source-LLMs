import pickle
import faiss
from sentence_transformers import SentenceTransformer


INDEX_PATH = "data/processed/index"


def load_index():
    index = faiss.read_index(f"{INDEX_PATH}.faiss")
    with open(f"{INDEX_PATH}_chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def retrieve_chunks(query, model, index, chunks, k=3):
    query_vec = model.encode([query], convert_to_numpy=True)
    _, indices = index.search(query_vec, k)
    return [chunks[i] for i in indices[0]]
