from dotenv import load_dotenv
import gradio as gr
from litellm import  completion
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

load_dotenv(override=True)


OLLAMA_BASE="http://localhost:11434"
qwenModel: str = "ollama_chat/qwen3:8b"
llamaModel: str = "ollama_chat/llama3.2"
gemmaModel: str = "ollama_chat/gemma3:270m"


def fetchWebsiteContent(url: str, headless: bool = True, timeoutMs: int = 60_000) -> str:
    """
    Open the url in a real browser and return the page as plain text:
    title, meta description, all visible text and every link on the page.
    """
    # Runs inside the page: returns visible text and all links with absolute urls
    extractPageJs = """
    () => {
        const links = [];
        for (const a of document.querySelectorAll('a[href]')) {
            const href = a.href;  // already resolved to an absolute url
            if (!href || href.startsWith('javascript:')) continue;
            const text = (a.innerText || a.getAttribute('aria-label') || a.title || '').trim().replace(/\\s+/g, ' ');
            links.push({ text, href });
        }

        return { text: document.body.innerText, links };
    }
    """

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=headless)
        page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=timeoutMs)
        try:
            page.wait_for_load_state("networkidle", timeout=10_000)
        except PlaywrightTimeout:
            pass  # busy sites never go fully idle; use what has loaded

        title = page.title()
        description = page.locator("meta[name='description']").first
        description = description.get_attribute("content") if description.count() else ""
        data = page.evaluate(extractPageJs)
        browser.close()

    # Drop empty and duplicate lines
    lines = [line.strip() for line in data["text"].splitlines()]
    text = "\n".join(dict.fromkeys(line for line in lines if line))

    # Drop duplicate urls and in-page anchors like "#top"
    links = {}
    for link in data["links"]:
        href = link["href"].split("#")[0]
        if href and href not in links:
            links[href] = link["text"] or "(no text)"
    linksText = "\n".join(f"- {linkText}: {href}" for href, linkText in links.items())

    return (
        f"URL: {url}\nTitle: {title}\nDescription: {description or 'not provided'}\n\n"
        f"## Content\n{text}\n\n## Links\n{linksText}"
    )



def callQwen(prompt: str):
    systemPrompt = """You are a professional business analyst and copywriter.
            You receive raw text scraped from a company's homepage. It may contain navigation menus,
            cookie notices, footer links and other noise. Ignore that noise.

            Your task: create a polished company brochure in Markdown for potential business clients and investors.

            Structure:
            # <Company name>
            ## Overview
            ## What They Offer
            ## Who They Serve
            ## Why Choose Them
            ## Contact & Next Steps

            Behaviour:
            - Professional, confident, analytical tone.
            - Highlight concrete value: services, industries, results, numbers, clients, if present.
            - Use only facts found in the content. Never invent clients, numbers, awards or contact details.
            - If a section has no supporting information, omit it.
            - Write the brochure in the same language as the website content.
            - Output only the brochure, no commentary. /no_think
        """

    messages = [
        {"role": "system", "content": systemPrompt},
        {"role": "user", "content": prompt},
    ]

    llmRes = completion(
        model=qwenModel,
        messages=messages,
        api_base = OLLAMA_BASE
        # stream=True
    )

    return llmRes.choices[0].message.content

def callLlama(prompt: str):
    systemPrompt = """You are a creative marketing copywriter.
            You receive raw text scraped from a company's homepage. Ignore menus, cookie notices,
            footers and other website noise.

            Your task: write an engaging company brochure in Markdown for customers and future employees.

            Structure:
            # <Company name>
            A catchy one-line tagline
            ## Who We Are
            ## What We Do
            ## Our Culture & Values
            ## Get in Touch

            Behaviour:
            - Warm, friendly, energetic tone. Speak to the reader as "you".
            - Keep paragraphs short and easy to read.
            - Use only facts from the content. Do not make anything up.
            - Skip any section the content does not support.
            - Write in the same language as the website content.
            - Output only the brochure.
        """

    messages = [
        {"role": "system", "content": systemPrompt},
        {"role": "user", "content": prompt},
    ]

    llmRes = completion(
        model=llamaModel,
        messages=messages,
        api_base = OLLAMA_BASE
        # stream=True
    )
    return llmRes.choices[0].message.content

def callGemma(prompt):
    sysPrompt = """Write a short company brochure in Markdown from the website text below.
        Use only facts from the text. Ignore menus and cookie notices.

        Format:
        # Company name
        One sentence about the company.
        - 3 to 5 bullet points about what they offer
        Contact information if present.
    """

    messages = [
        {"role": "system", "content": sysPrompt},
        {"role": "user", "content": prompt},
    ]

    llmRes = completion(
        model=gemmaModel,
        messages=messages,
        api_base = OLLAMA_BASE
        # stream=True
    )
    return llmRes.choices[0].message.content

def callLLM(model: str, companyName: str, companyUrl: str):
    prompt = f"Company name: {companyName}\n\nWebsite content:\n{fetchWebsiteContent(companyUrl)}"
    if model == "qwen":
        result = callQwen(prompt)
    elif model == "llama":
        result = callLlama(prompt)
    elif model == "gemma":
        result = callGemma(prompt)
    else:
        raise RuntimeError("Wrong type of model", model)

    return result



modelSelector = gr.Dropdown(label="Model", choices=["qwen", "llama", "gemma"], value="qwen")
companyNameTextbox = gr.Textbox(label="Company name",lines=1, placeholder="Company Name")
companyUrlTextBox = gr.Textbox(label="Company Url", placeholder="Company name")
outputTextbox = gr.Markdown(label="Response")


gr.Interface(inputs=[modelSelector, companyNameTextbox, companyUrlTextBox],
             outputs=[outputTextbox],
             fn=callLLM,
             title="Brochure Creator",
             description="This is a brochure creator using Gradio and OpenAI API.").launch(inbrowser=True)



