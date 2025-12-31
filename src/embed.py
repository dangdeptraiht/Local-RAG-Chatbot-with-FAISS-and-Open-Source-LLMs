import pickle
import faiss
from sentence_transformers import SentenceTransformer
from src.ingest import load_text, clean_text, chunk_text


DATA_PATH = "data/raw/hf_transformers.txt"
OUT_PATH = "data/processed/index"


def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def embed_chunks(model, chunks):
    return model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)


def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def save_index(index, chunks):
    faiss.write_index(index, f"{OUT_PATH}.faiss")
    with open(f"{OUT_PATH}_chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)


if __name__ == "__main__":
    text = load_text(DATA_PATH)
    text = clean_text(text)
    chunks = chunk_text(text)

    print(f"Embedding {len(chunks)} chunks...")

    model = load_embedding_model()
    embeddings = embed_chunks(model, chunks)

    index = build_faiss_index(embeddings)
    save_index(index, chunks)

    print("Index saved.")
