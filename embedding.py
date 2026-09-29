import requests
import math

texts = [
    "Python es un lenguaje de programación.",
    "Python permite desarrollar software y aplicaciones.",
    "Me gusta comer pizza los fines de semana."
]


def get_embedding(text):
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={
            "model": "nomic-embed-text",
            "prompt": text
        }
    )

    return response.json()["embedding"]


def cosine_similarity(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))

    magnitude_a = math.sqrt(sum(x * x for x in a))
    magnitude_b = math.sqrt(sum(x * x for x in b))

    return dot_product / (magnitude_a * magnitude_b)


embeddings = [get_embedding(text) for text in texts]

for i in range(len(texts)):
    for j in range(i + 1, len(texts)):
        similarity = cosine_similarity(
            embeddings[i],
            embeddings[j]
        )

        print(
            f"\n'{texts[i]}'"
            f"\nvs."
            f"\n'{texts[j]}'"
        )

        print(f"Similitud: {similarity:.4f}")