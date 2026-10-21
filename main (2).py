import os
import textwrap
from datetime import datetime

import gradio as gr
from openai import OpenAI, OpenAIError

# The client reads your key from the OPENAI_API_KEY environment variable.
client = OpenAI()

# Set OPENAI_MODEL to switch models without editing code.
MODEL = os.getenv("OPENAI_MODEL", "gpt-6-luna")

OUTPUT_DIR = "outputs"
CHUNK_CHARS = 4000

SUMMARY_STYLES = [
    "Convert the content into bullet points.",
    "Write an executive summary of the content.",
    "Shorten the content without losing any information.",
]


def save_file(content, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as outfile:
        outfile.write(content)


def generate_summary(chunk, max_words, style):
    response = client.responses.create(
        model=MODEL,
        instructions=(
            "You are a skilled writer who summarizes content while keeping all key information. "
            f"{SUMMARY_STYLES[style]} Use about {max_words} words or fewer."
        ),
        input=chunk,
    )
    return response.output_text.strip()


def chunk_summarize(all_text, max_words, style):
    if not all_text or not all_text.strip():
        return "Please enter some text to summarize."
    if style is None:
        style = 0

    chunks = textwrap.wrap(all_text, CHUNK_CHARS)
    results = []

    try:
        for count, chunk in enumerate(chunks, start=1):
            summary = generate_summary(chunk, int(max_words), style)
            print(f"\n\n{count} of {len(chunks)}\n{summary}")
            results.append(summary)
    except OpenAIError as error:
        return f"OpenAI request failed: {error}"

    full_summary = "\n\n".join(results)

    # Save a copy for reference, then return the full summary.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_file(full_summary, os.path.join(OUTPUT_DIR, f"output_{timestamp}.txt"))

    return full_summary


def to_gradio():
    demo = gr.Interface(
        fn=chunk_summarize,
        inputs=[
            gr.Textbox(lines=8, placeholder="Enter content to summarize...", label="Content"),
            gr.Slider(minimum=50, maximum=750, value=250, step=10, label="Target words per chunk"),
            gr.Dropdown(["bullets", "executive", "trim"], type="index", value="bullets", label="Summary style"),
        ],
        outputs=gr.Textbox(label="Summary"),
        title="Prompt Engineering Summarization Bot",
    )
    demo.launch(debug=True, share=True)


if __name__ == "__main__":
    to_gradio()
