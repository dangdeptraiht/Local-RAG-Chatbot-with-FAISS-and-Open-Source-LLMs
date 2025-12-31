def build_prompt(context_chunks, question):
    context = "\n\n".join(context_chunks)

    return f"""
You are an assistant that answers questions ONLY using the provided context.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
""".strip()
