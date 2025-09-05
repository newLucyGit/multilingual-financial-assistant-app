import os
import requests
from dotenv import load_dotenv

load_dotenv()  # loads .env in the project root

API_KEY = os.getenv("MISTRAL_API_KEY")
assert API_KEY, "Missing MISTRAL_API_KEY in .env"

# Choose a chat model you have access to from your Mistral dashboard.
MODEL = "open-mistral-7b"  # if this errors, switch to another available chat model

url = "https://api.mistral.ai/v1/chat/completions"
headers = {"Authorization": f"Bearer {API_KEY}"}
payload = {
    "model": MODEL,
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello in English and Hindi."}
    ]
}

resp = requests.post(url, headers=headers, json=payload, timeout=60)
resp.raise_for_status()
data = resp.json()

print(data["choices"][0]["message"]["content"])
