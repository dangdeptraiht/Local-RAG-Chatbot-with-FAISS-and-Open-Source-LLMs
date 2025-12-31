from src.rag_engine import RAGEngine


def main():
    rag = RAGEngine()

    while True:
        q = input("\nAsk a question (or 'exit'): ")
        if q.lower() == "exit":
            break
        print("\nAnswer:\n", rag.ask(q))


if __name__ == "__main__":
    main()
