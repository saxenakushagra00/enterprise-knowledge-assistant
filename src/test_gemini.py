import os
import time

from google import genai
from google.genai import types

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"],
    http_options=types.HttpOptions(timeout=30000),
)

# >>> PASTE YOUR MODEL NAMES HERE (this is the only part you edit) <<<
MODELS = [
    "gemini-3.5-flash",
    "gemini-flash-lite-latest",
    "gemini-3.1-flash-lite-preview",
    "gemini-3-flash-preview",
]


def ask_gemini(prompt: str, retries: int = 4) -> str:
    for attempt in range(retries):
        for model in MODELS:
            try:
                response = client.models.generate_content(model=model, contents=prompt)
                return response.text
            except Exception as e:
                print(f"Attempt {attempt + 1} with {model} failed: {type(e).__name__}")
        time.sleep(2 ** attempt * 2)
    return "Gemini is busy right now. Please try again in a minute."


if __name__ == "__main__":
    print(ask_gemini("Say hello in one short sentence."))