import sys
from urllib.parse import urljoin

from dotenv import load_dotenv
from openai import OpenAI
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

load_dotenv(override=True)
openai_client = OpenAI()


response = openai_client.chat.completions.create(model="gpt-4.1-nano",messages = [
    {"role":"user","content":"Tell me a fun fact."}
])

print(response.choices[0].message.content)