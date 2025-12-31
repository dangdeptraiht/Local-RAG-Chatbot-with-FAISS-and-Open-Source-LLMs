from pathlib import Path


def load_text(file_path: str) -> str:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"{file_path} not found")

    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    return " ".join(text.split())


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100):
    words = text.split()
    chunks = []

    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = words[i:i + chunk_size]
        if len(chunk) < 50:
            continue
        chunks.append(" ".join(chunk))

    return chunks
