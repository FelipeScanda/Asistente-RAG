import requests
from embedding import search


OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "qwen3:8b"


def generate_answer(query, documents):

    context = "\n\n".join(
        document[2]
        for document in documents
    )

    prompt = f"""
You are a question-answering assistant.

Answer the user's question using only the information
contained in the context below.

If the context does not contain enough information to
answer the question, say that the available context
does not contain enough information.

Context:
{context}

Question:
{query}

Answer:
"""

    response = requests.post(
        OLLAMA_GENERATE_URL,
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False
        }
    )

    response.raise_for_status()

    return response.json()["response"]


def main():

    query = input("Pregunta: ")

    documents = search(query, k=5)

    print("\nDocumentos recuperados:")
    print("-----------------------")

    for rs_id, gene, content, distance in documents:
        print(f"rs{rs_id} | distancia: {distance:.4f}")
        print(content)
        print()

    answer = generate_answer(query, documents)

    print("\nRespuesta:")
    print("---------")
    print(answer)


if __name__ == "__main__":
    main()