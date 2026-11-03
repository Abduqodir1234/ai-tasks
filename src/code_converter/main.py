import time

import gradio as gr

from converter import convertCodeStream, extractCppCode, models, runCpp, runPython


def convert(sourceCode: str, modelName: str):
    if not sourceCode.strip():
        yield "", "", "Paste some code first"
        return

    start = time.perf_counter()
    response = ""
    try:
        for response in convertCodeStream(sourceCode, models[modelName]):
            yield response, extractCppCode(response), f"Converting… {time.perf_counter() - start:.1f}s"
    except Exception as e:
        yield response, extractCppCode(response), f"Error from {modelName}: {e}"
        return

    yield response, extractCppCode(response), f"Converted by {modelName} in {time.perf_counter() - start:.2f}s"


with gr.Blocks(title="Python → C++ Converter") as ui:
    gr.Markdown("## Python → C++ converter\nPaste Python code, convert it to fast C++, then run both and compare.")

    with gr.Row():
        model = gr.Dropdown(list(models), value=next(iter(models)), label="Model")
        args = gr.Textbox(label="Program arguments (used for both runs)", placeholder="e.g. 0.5")

    with gr.Row():
        with gr.Column():
            sourceCode = gr.Code(label="Python code", language="python", lines=25)
            convertButton = gr.Button("Convert to C++", variant="primary")
        with gr.Column():
            outputCode = gr.Code(label="C++ code", language="cpp", lines=25)
            convertTime = gr.Markdown()

    with gr.Accordion("Full model response (build command, optimizations, behavior notes)", open=False):
        fullResponse = gr.Markdown()

    with gr.Row():
        with gr.Column():
            runSourceButton = gr.Button("Run Python code")
            sourceTime = gr.Markdown()
            sourceOutput = gr.Textbox(label="Python output", lines=12)
        with gr.Column():
            runOutputButton = gr.Button("Run C++ code")
            outputTime = gr.Markdown()
            cppOutput = gr.Textbox(label="C++ output", lines=12)

    convertButton.click(convert, [sourceCode, model], [fullResponse, outputCode, convertTime])
    runSourceButton.click(runPython, [sourceCode, args], [sourceOutput, sourceTime])
    runOutputButton.click(runCpp, [outputCode, args], [cppOutput, outputTime])


if __name__ == "__main__":
    ui.launch()
