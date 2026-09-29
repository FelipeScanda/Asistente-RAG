import requests

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "qwen3:8b",
        "prompt": "¿Qué variantes están asociadas al gen LPL?",
        "stream": False
    }
)

data = response.json()

print(data["response"])