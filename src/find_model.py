import os

from google import genai
from google.genai import types

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"],
    http_options=types.HttpOptions(timeout=15000),
)

SKIP = ("tts", "image", "live", "audio", "embedding", "robotics", "computer")

names = [m.name.replace("models/", "") for m in client.models.list()]
candidates = [
    n for n in names
    if "gemini" in n and "flash" in n and not any(word in n for word in SKIP)
]
print("Flash models found:", len(candidates))

for name in candidates:
    try:
        r = client.models.generate_content(model=name, contents="Say hi.")
        print("OK     ", name, "->", r.text.strip()[:30])
    except Exception as e:
        print("FAILED ", name, type(e).__name__, str(e)[:50])