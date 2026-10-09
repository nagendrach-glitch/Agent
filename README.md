---
title: Healthcare Multi RAG Assistant
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---

# 🏥 Healthcare Multi-RAG Assistant

Routes each question to one of four FAISS knowledge bases (Medical documents, Drug CSV,
Symptoms JSON, WHO guidelines) and answers with Groq Llama 3.

**Required secret:** `GROQ_API_KEY` (Space → Settings → Variables and secrets).

> ⚠️ Educational demo only. Not medical advice — consult a qualified doctor.

## Adding data
Put any number of files directly in `data/` (no subfolders needed):
- `.csv` → Drug RAG
- `.json` → Symptom RAG
- `.txt` / `.md` / `.pdf` → Medical RAG (filename containing `who` → WHO RAG)

Commit the files; the Space restarts and rebuilds the indexes automatically.
