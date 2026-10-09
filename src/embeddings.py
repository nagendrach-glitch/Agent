from langchain_huggingface import HuggingFaceEmbeddings


def get_embeddings():
    # Force CPU: on ZeroGPU Spaces, touching CUDA outside a @spaces.GPU
    # function raises "Low-level CUDA init" errors. MiniLM is fast on CPU anyway.
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
