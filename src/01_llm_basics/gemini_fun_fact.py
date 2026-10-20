import sys
from urllib.parse import urljoin
import os

from dotenv import load_dotenv
from openai import OpenAI
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

load_dotenv(override=True)
geminiKey = os.environ.get("GEMINI_API_KEY")
gemini_client = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/",api_key=geminiKey)

print(gemini_client.models.list())


response = gemini_client.chat.completions.create(model="gemini-3.1-pro-preview",messages = [
    {"role":"user","content":"Tell me a fun fact."}
])

print(response.choices[0].message.content)