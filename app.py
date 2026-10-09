import gradio as gr

from src.rag_chain import ask_question

# ZeroGPU Spaces require at least one @spaces.GPU function at startup.
# This app runs fine on CPU, so this is a tiny placeholder that satisfies the check.
try:
    import spaces

    @spaces.GPU
    def _gpu_placeholder():
        return "ok"
except ImportError:  # running locally / on CPU hardware
    pass


def respond(question):
    if not question or not question.strip():
        return "", "Please enter a question."
    try:
        result = ask_question(question)
        return result["rag_used"], result["answer"]
    except Exception as e:
        return "", f"Error: {e}"


with gr.Blocks(title="Healthcare Multi RAG Assistant") as demo:
    gr.Markdown(
        "# 🏥 Healthcare Multi-RAG Assistant\n"
        "Educational demo — not medical advice. Consult a doctor."
    )
    question = gr.Textbox(label="Ask a healthcare question",
                          placeholder="e.g. What are symptoms of malaria?")
    btn = gr.Button("Ask", variant="primary")
    rag = gr.Textbox(label="RAG selected", interactive=False)
    answer = gr.Textbox(label="Answer", lines=8, interactive=False)

    gr.Examples(
        examples=[
            "Explain dengue disease",
            "What are symptoms of malaria?",
            "Which medicine is used for fever?",
            "What does WHO recommend for mosquito prevention?",
        ],
        inputs=question,
    )

    btn.click(respond, question, [rag, answer])
    question.submit(respond, question, [rag, answer])

if __name__ == "__main__":
    demo.launch()
