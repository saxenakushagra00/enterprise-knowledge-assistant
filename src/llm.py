import os
import time

from google import genai
from google.genai import types

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"],
    http_options=types.HttpOptions(timeout=30000),
)

MODELS = ["gemini-3.5-flash", "gemini-3.6-flash", "gemini-3.1-flash-lite", "gemini-3.5-flash-lite"]


def ask_gemini(prompt: str, retries: int = 3) -> str:
    for attempt in range(retries):
        for model in MODELS:
            try:
                response = client.models.generate_content(model=model, contents=prompt)
                return response.text
            except Exception as e:
                print(f"Attempt {attempt + 1} with {model} failed: {type(e).__name__}")
        time.sleep(2 ** attempt * 2)
    return "Gemini is busy right now. Please try again in a minute."