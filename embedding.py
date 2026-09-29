import requests
import psycopg

OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL = "nomic-embed-text"


def get_embedding(text):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": text
        }
    )

    response.raise_for_status()

    return response.json()["embedding"]


def search(query, k=2):

    query_embedding = get_embedding(query)

    with psycopg.connect("dbname=rag_demo") as conn:

        with conn.cursor() as cur:

            cur.execute(
                """
                SELECT
                    rs_id,
                    gene,
                    content,
                    embedding <=> %s::vector AS distance
                FROM variants
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (
                    query_embedding,
                    query_embedding,
                    k
                )
            )

            return cur.fetchall()


if __name__ == "__main__":

    query = input("Pregunta: ")

    results = search(query, k=2)

    print("\nDocumentos recuperados:\n")

    for rs_id, gene, content, distance in results:

        print(f"rs{rs_id} | distancia: {distance:.4f}")
        print(content)
        print()