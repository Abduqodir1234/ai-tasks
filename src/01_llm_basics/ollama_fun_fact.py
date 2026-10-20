import sys
from urllib.parse import urljoin
import os

from dotenv import load_dotenv
from openai import OpenAI
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

load_dotenv(override=True)
ollamaApiKey = "Ollama"
ollamaBaseUrl = 'http://localhost:11434/v1'

ollamaClient = OpenAI(base_url=ollamaBaseUrl,api_key=ollamaApiKey);

response = ollamaClient.chat.completions.create(model="llama3.2",messages=[{"role":"user","content":"Tell me a fun fact."}])

print(response.choices[0].message.content)