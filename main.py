import requests

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "qwen3:8b",
        "prompt": "Explica qué es un RAG en una sola frase.",
        "stream": False
    }
)

data = response.json()

print(data["response"])