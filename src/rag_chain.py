from pathlib import Path

from src.router import route_question
from src.retrievers import load_retrievers, FAISS_DIR
from src.llm import get_llm

_retrievers = None
_llm = None


def _init():
    global _retrievers, _llm
    if _retrievers is None:
        if not (FAISS_DIR / "medical_index" / "index.faiss").exists():
            from src.vector_store import create_indexes
            create_indexes()
        _retrievers = load_retrievers()
    if _llm is None:
        _llm = get_llm()


def ask_question(question):
    _init()

    rag_type = route_question(question)
    docs = _retrievers[rag_type].invoke(question)
    context = "\n".join(d.page_content for d in docs)

    prompt = f"""You are a healthcare assistant.
Use only the context below. If the answer is not in the context, say you don't have that information.
Remind the user to consult a doctor for medical decisions.

Context:
{context}

Question:
{question}

Answer:
"""
    response = _llm.invoke(prompt)
    return {"rag_used": rag_type, "answer": response.content}
