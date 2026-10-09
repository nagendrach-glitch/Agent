from pathlib import Path

from langchain_community.vectorstores import FAISS

from src.embeddings import get_embeddings

BASE_DIR = Path(__file__).resolve().parent.parent
FAISS_DIR = BASE_DIR / "faiss_indexes"

INDEXES = {
    "medical": "medical_index",
    "drug": "drug_index",
    "symptom": "symptom_index",
    "who": "who_index",
}


def load_retrievers():
    embeddings = get_embeddings()
    retrievers = {}
    for name, folder in INDEXES.items():
        db = FAISS.load_local(
            str(FAISS_DIR / folder),
            embeddings,
            allow_dangerous_deserialization=True,  # indexes are built by us, not user-supplied
        )
        retrievers[name] = db.as_retriever(search_kwargs={"k": 3})
    return retrievers
