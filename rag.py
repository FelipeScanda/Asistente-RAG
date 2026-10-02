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
Eres un asistente inteligente que explica preguntas de genomica al usuario.

Debes contestar la pregunta del usuario usando solo la informacion que está en el texto de abajo.

Si el contexto no contiene la información suficiente para responder la pregunta, di que el contecto no contiene la información suficiente.

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