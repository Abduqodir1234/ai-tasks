import random
from pathlib import Path

import gradio as gr
import numpy as np
import plotly.graph_objects as go
from sklearn.manifold import TSNE

from constants.assistant_constant import VECTOR_DB_NAME
from utils.get_vector_db import getDb

# Same absolute path as ingest.py and llm_as_controller.py
DB_NAME = str(Path(__file__).parent.parent / VECTOR_DB_NAME)
# Stored vectors are read as they are, so no embedding model has to be loaded
db = getDb(DB_NAME, None)
collection = db._collection

SECTION_COLORS = {"docs": "#2f6fdb", "handbook": "#e07b20"}
OTHER_COLOR = "#8a8a8a"
# t-SNE on a few thousand 1024-dim vectors takes seconds; on the whole db it takes minutes
MAX_SAMPLE_SIZE = 5000


def getStats() -> str:
    """Markdown summary: vector count, dimensions and chunks per section."""
    count = collection.count()
    if count == 0:
        return f"**{DB_NAME}** is empty. Run `ingest.py` first."

    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    sections = collection.get(include=["metadatas"])["metadatas"]
    perSection: dict[str, int] = {}
    for metadata in sections:
        section = (metadata or {}).get("section", "unknown")
        perSection[section] = perSection.get(section, 0) + 1

    lines = [
        f"**Database:** `{DB_NAME}`",
        f"**Collection:** `{collection.name}`",
        f"**Vectors:** {count:,} with {len(sample_embedding):,} dimensions",
        "**Per section:** " + ", ".join(f"{name}: {total:,}" for name, total in sorted(perSection.items())),
    ]
    return "\n\n".join(lines)


def sampleVectors(sampleSize: int, section: str):
    """Pick up to sampleSize random chunks (optionally from one section) with their vectors."""
    where = {"section": section} if section != "all" else None
    ids = collection.get(where=where, include=[])["ids"]
    if len(ids) > sampleSize:
        ids = random.Random(42).sample(ids, sampleSize)
    if not ids:
        return np.empty((0, 0)), [], []

    data = collection.get(ids=ids, include=["embeddings", "documents", "metadatas"])
    return np.array(data["embeddings"]), data["documents"], data["metadatas"]


def plotVectors(sampleSize: int, section: str, dimensions: str) -> go.Figure:
    """Reduce the sampled vectors with t-SNE and draw them, one trace per section."""
    vectors, documents, metadatas = sampleVectors(int(sampleSize), section)
    if len(vectors) < 3:
        return go.Figure().update_layout(title="Not enough vectors to plot")

    components = 3 if dimensions == "3D" else 2
    # perplexity must be below the number of samples
    tsne = TSNE(n_components=components, random_state=42, perplexity=min(30, len(vectors) - 1))
    reduced = tsne.fit_transform(vectors)

    sections = [(metadata or {}).get("section", "unknown") for metadata in metadatas]
    hoverTexts = [
        f"Section: {s}<br>Title: {(m or {}).get('title', '')}<br>Source: {(m or {}).get('source', '')}<br>"
        f"Text: {d[:150].replace(chr(10), ' ')}..."
        for s, m, d in zip(sections, metadatas, documents)
    ]

    fig = go.Figure()
    for name in sorted(set(sections)):
        index = [i for i, s in enumerate(sections) if s == name]
        marker = dict(size=4 if components == 3 else 5, color=SECTION_COLORS.get(name, OTHER_COLOR), opacity=0.8)
        text = [hoverTexts[i] for i in index]
        if components == 3:
            fig.add_trace(go.Scatter3d(
                x=reduced[index, 0], y=reduced[index, 1], z=reduced[index, 2],
                mode="markers", name=f"{name} ({len(index):,})", marker=marker, text=text, hoverinfo="text",
            ))
        else:
            fig.add_trace(go.Scatter(
                x=reduced[index, 0], y=reduced[index, 1],
                mode="markers", name=f"{name} ({len(index):,})", marker=marker, text=text, hoverinfo="text",
            ))

    fig.update_layout(
        title=f"{dimensions} Chroma Vector Store Visualization ({len(vectors):,} chunks)",
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"),
        height=700,
        margin=dict(r=20, b=10, l=10, t=40),
    )
    return fig


with gr.Blocks(title="Vector DB Explorer") as demo:
    gr.Markdown("# Vector DB Explorer\nRandom sample of stored chunks, reduced to 2D or 3D with t-SNE. Hover a point to see its text.")
    stats = gr.Markdown(getStats())
    with gr.Row():
        sampleSize = gr.Slider(100, MAX_SAMPLE_SIZE, value=1500, step=100, label="Sample size")
        section = gr.Radio(["all", "docs", "handbook"], value="all", label="Section")
        dimensions = gr.Radio(["2D", "3D"], value="2D", label="Dimensions")
    plotButton = gr.Button("Plot", variant="primary")
    plot = gr.Plot()

    plotButton.click(plotVectors, inputs=[sampleSize, section, dimensions], outputs=plot)
    plotButton.click(getStats, outputs=stats)


if __name__ == "__main__":
    demo.launch(inbrowser=True)
