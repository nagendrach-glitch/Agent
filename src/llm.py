import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

DEFAULT_MODEL = "openai/gpt-oss-20b"


def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it in Space Settings → "
            "Variables and secrets."
        )
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", DEFAULT_MODEL),
        temperature=0,
        api_key=api_key,
    )