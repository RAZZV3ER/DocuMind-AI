import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv(override=True)

key = os.getenv("GROQ_API_KEY")

print("Key loaded:", bool(key))
print("Key length:", len(key) if key else 0)

if not key:
    raise RuntimeError("GROQ_API_KEY was not loaded from .env")

client = Groq(api_key=key)

try:
    print("Available models:", len(client.models.list().data))
    print("Groq authentication successful!")
except Exception as e:
    print(type(e).__name__, str(e))