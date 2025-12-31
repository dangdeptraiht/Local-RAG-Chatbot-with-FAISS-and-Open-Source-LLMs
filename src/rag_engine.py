from src.retriever import load_index, load_embedding_model, retrieve_chunks
from src.prompt import build_prompt
from src.llm import load_llm, generate_answer


class RAGEngine:
    def __init__(self):
        self.embed_model = load_embedding_model()
        self.index, self.chunks = load_index()
        self.tokenizer, self.llm = load_llm()

    def ask(self, question: str) -> str:
        context = retrieve_chunks(
            question,
            self.embed_model,
            self.index,
            self.chunks
        )

        prompt = build_prompt(context, question)
        return generate_answer(prompt, self.tokenizer, self.llm)
