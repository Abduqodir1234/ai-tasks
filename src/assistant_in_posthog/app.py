import gradio as gr

from dtos.llm_dto import GradioChatHistory
from llm_as_controller import main


def gradioToChatHistory(gradioHistory: list[dict]) -> list[GradioChatHistory]:
    """
    Turn Gradio 6 history into the {role, content: str} messages llm.main expects.
    Gradio stores content as a list of parts ({"type": "text", "text": ...}) and adds
    metadata/options keys that convert_to_messages and combined_question can't use.
    """
    history: list[GradioChatHistory] = []
    for message in gradioHistory:
        content = message["content"]
        if isinstance(content, list):
            content = "\n".join(
                part["text"] for part in content if part.get("type") == "text" and part.get("text")
            )
        if not content:
            continue  # skip file-only or empty messages
        history.append({"role": message["role"], "content": content})
    return history


def chat(prompt: str, gradioHistory: list[dict]) -> str:
    try:
        reply = main(prompt, gradioToChatHistory(gradioHistory))
    except Exception as e:
        # Show the error in the chat instead of Gradio's generic "Error" toast
        print(f"Chat failed: {type(e).__name__}: {e}")
        return f"Sorry, something went wrong: {type(e).__name__}: {e}"
    return reply if isinstance(reply, str) else str(reply)


gr.ChatInterface(
    fn=chat,
    title="PostHog Assistant",
    description="Ask about PostHog docs and handbook. Answers come from the ingested pages.",
    examples=[
        "How do I set up feature flags in a Next.js app?",
        "What is the difference between product analytics and web analytics?",
        "Why does PostHog exist? What is its mission?",
    ],
).launch(inbrowser=True)
