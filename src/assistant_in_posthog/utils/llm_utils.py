import re
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder

from constants.assistant_constant import (
    OPTIMIZE_DOC_PART_USER_PROMPT,
    OPTIMIZE_DOC_SYSTEM_PROMPT,
    OPTIMIZE_DOC_USER_PROMPT,
    REWRITE_PROMPT_SYSTEM_PROMPT,
    REWRITE_PROMPT_USER_PROMPT,
)
from dtos.llm_dto import GradioChatHistory


class RewrittenPrompt(BaseModel):
    prompt: str = Field(description="The latest message rewritten to stand alone, with any relevant context from history")


class OptimizedPage(BaseModel):
    content: str = Field(description="The whole page rewritten in Markdown, starting with a '# <title>' heading")


def combined_question(question: str, history: list[GradioChatHistory] = []) -> str:
    """
    Combine all the user's messages into a single string.
    """
    prior = "\n".join(m["content"] for m in history if m["role"] == "user")
    return prior + "\n" + question


def rerank_context(question: str, docs: list[Document], reranker: CrossEncoder) -> list[Document]:
    """
    Score every doc against question with the cross-encoder reranker and return
    them most relevant first. If scoring fails, the original order is returned.
    """
    if len(docs) < 2:
        return docs

    try:
        ranked = reranker.rank(question, [doc.page_content for doc in docs])
    except Exception as e:
        print(f"Rerank failed, keeping retriever order: {type(e).__name__}: {e}")
        return docs

    order = [int(r["corpus_id"]) for r in ranked]
    print("Rerank order:", [(id + 1, round(float(r["score"]), 3)) for id, r in zip(order, ranked)])
    return [docs[id] for id in order]


def improve_current_prompt(prompt: str, history: list[GradioChatHistory], llm: BaseChatModel) -> str:
    """
    Ask llm to improve the prompt by adding context from history.
    llm must support with_structured_output. If the call fails or the answer is
    empty, the prompt is returned as given.
    """

    prior = "\n".join(m["content"] for m in history if m["role"] == "user")
    messages = [
        SystemMessage(content=REWRITE_PROMPT_SYSTEM_PROMPT),
        HumanMessage(content=REWRITE_PROMPT_USER_PROMPT.format(prior_messages=prior, current_prompt=prompt)),
    ]
    try:
        result = llm.with_structured_output(RewrittenPrompt).invoke(messages)
        rewritten = RewrittenPrompt.model_validate(result).prompt.strip()
    except Exception as e:
        print(f"Rewrite failed, keeping original prompt: {type(e).__name__}: {e}")
        return prompt
    print("Rewritten prompt:", rewritten)
    return rewritten or prompt


# Parts share this many chars so a sentence cut at a part boundary is whole in one of them;
# the shared text is rewritten in both parts, so it shows up twice in the joined page
PAGE_PART_OVERLAP = 200
# Split points for pages longer than maxChars, tried in order: headings first so
# each part keeps whole sections, then paragraphs, lines and words
PAGE_PART_SEPARATORS = ["\n# ", "\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""]


def optimize_document(content: str, title: str | None, llm: BaseChatModel, maxChars: int = 40000, minRatio: float = 0.5) -> str:
    """
    Ask llm to rewrite a whole page so its chunks are easier to retrieve:
    descriptive headings, sections that make sense alone, schemas turned into
    sentences. See OPTIMIZE_DOC_SYSTEM_PROMPT. llm must support with_structured_output.

    A page longer than maxChars would not fit the model's context, so it is split
    into parts of at most maxChars - PAGE_PART_OVERLAP chars that overlap by
    PAGE_PART_OVERLAP, at headings where possible, and each part is rewritten on its own (see optimize_part), with the page's outline (see
    page_outline) so the model knows what the whole page is about. Parts after
    the first lose the repeated "# <title>" heading the model starts every answer with.
    """
    if not content.strip():
        return content
    if len(content) <= maxChars:
        return optimize_part(content, title, llm, minRatio)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=maxChars - PAGE_PART_OVERLAP,
        chunk_overlap=PAGE_PART_OVERLAP,
        separators=PAGE_PART_SEPARATORS,
        keep_separator="start",
        length_function=len,
    )
    parts = splitter.split_text(content)
    print(f"Page {title!r} is {len(content)} chars, optimizing it in {len(parts)} parts")
    outline = page_outline(content, title)
    optimized = []
    for i, part in enumerate(parts):
        prompt = OPTIMIZE_DOC_PART_USER_PROMPT.format(
            title=title or "", outline=outline, content=part, part=i + 1, parts=len(parts)
        )
        rewritten = optimize_part(part, title, llm, minRatio, prompt)
        if i > 0:
            rewritten = strip_title_heading(rewritten, title)
        optimized.append(rewritten)
    return "\n\n".join(optimized)


def optimize_part(content: str, title: str | None, llm: BaseChatModel, minRatio: float, prompt: str | None = None) -> str:
    """
    Rewrite content, which must fit the model's context, with one llm call.
    prompt is the user message; without it content is sent as a whole page.
    The original content is returned when the call fails, or when the answer is
    shorter than minRatio of the original, which means the model summarized or
    cut the page instead of rewriting it.
    """
    messages = [
        SystemMessage(content=OPTIMIZE_DOC_SYSTEM_PROMPT),
        HumanMessage(content=prompt or OPTIMIZE_DOC_USER_PROMPT.format(title=title or "", content=content)),
    ]
    try:
        result = llm.with_structured_output(OptimizedPage).invoke(messages)
        optimized = OptimizedPage.model_validate(result).content
    except Exception as e:
        print(f"Optimize failed for {title!r}, keeping original: {type(e).__name__}: {e}")
        return content

    if len(optimized.strip()) < len(content) * minRatio:
        print(f"Optimize answer for {title!r} too short, keeping original")
        return content
    return optimized.strip()


HEADING_PATTERN = re.compile(r"^#{1,4} .+")
# A page with hundreds of headings (e.g. an API reference) would crowd out the part itself
MAX_OUTLINE_CHARS = 4000


def page_outline(content: str, title: str | None) -> str:
    """
    List the title and the #-#### headings of content, indented by level, skipping
    lines inside ``` code blocks (shell comments start with # too).
    Cut at MAX_OUTLINE_CHARS.
    """
    lines = [title] if title else []
    inCode = False
    for line in content.splitlines():
        if line.lstrip().startswith("```"):
            inCode = not inCode
        elif not inCode and HEADING_PATTERN.match(line) and line.lstrip("#").strip() != title:
            level = len(line) - len(line.lstrip("#"))
            lines.append("  " * (level - 1) + "- " + line.lstrip("#").strip())
    outline = "\n".join(lines)
    return outline if len(outline) <= MAX_OUTLINE_CHARS else outline[:MAX_OUTLINE_CHARS].rsplit("\n", 1)[0] + "\n- ..."


def strip_title_heading(content: str, title: str | None) -> str:
    """Drop a first line that is the "# <title>" heading, so a split page has it only once."""
    first, _, rest = content.partition("\n")
    if title and first.strip().lower() == f"# {title}".lower():
        return rest.lstrip("\n")
    return content
