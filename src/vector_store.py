"""Builds the 4 FAISS indexes from ALL files placed in the data/ folder.

Put files directly in data/ (subfolders also work). Routing by file:
  *.csv                      -> drug_index     (medicine tables)
  *.json                     -> symptom_index  (list of objects)
  *.txt / *.md / *.pdf       -> who_index      if filename contains "who"
                                medical_index  otherwise
"""
import json
from pathlib import Path

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.embeddings import get_embeddings

BASE_DIR = Path(__file__).resolve().parent.parent
DATA = BASE_DIR / "data"
OUT = BASE_DIR / "faiss_indexes"

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)


def _read_text(path):
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        return "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def _flat(item):
    """dict/list/scalar -> readable sentence."""
    if isinstance(item, dict):
        return ". ".join(f"{k}: {_flat(v)}" for k, v in item.items())
    if isinstance(item, list):
        return ", ".join(_flat(v) for v in item)
    return str(item)


def _save(docs, embeddings, name):
    if not docs:  # keep the app working even if a category has no files
        docs = [Document(page_content="No information available.",
                         metadata={"source": "empty"})]
    FAISS.from_documents(docs, embeddings).save_local(str(OUT / name))
    print(f"{name}: {len(docs)} chunks")


def create_indexes():
    embeddings = get_embeddings()
    medical, drug, symptom, who = [], [], [], []

    for path in sorted(p for p in DATA.rglob("*") if p.is_file()):
        ext = path.suffix.lower()
        meta = {"source": path.name}

        if ext == ".csv":
            df = pd.read_csv(path).dropna(how="all")
            for _, row in df.iterrows():
                drug.append(Document(page_content=_flat(row.dropna().to_dict()),
                                     metadata=meta))

        elif ext == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            for item in (data if isinstance(data, list) else [data]):
                symptom.append(Document(page_content=_flat(item), metadata=meta))

        elif ext in (".txt", ".md", ".pdf"):
            chunks = splitter.split_text(_read_text(path))
            target = who if "who" in path.stem.lower() else medical
            target.extend(Document(page_content=c, metadata=meta) for c in chunks)

    _save(medical, embeddings, "medical_index")
    _save(drug, embeddings, "drug_index")
    _save(symptom, embeddings, "symptom_index")
    _save(who, embeddings, "who_index")


if __name__ == "__main__":
    create_indexes()
    print("FAISS indexes created successfully")
