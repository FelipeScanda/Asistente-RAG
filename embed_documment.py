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


with psycopg.connect("dbname=rag_demo") as conn:

    with conn.cursor() as cur:

        cur.execute("""
            SELECT id, rs_id, content
            FROM variants
            WHERE embedding IS NULL
        """)

        rows = cur.fetchall()

        print(f"Documentos pendientes: {len(rows)}")

        for row_id, rs_id, content in rows:

            print(f"Generando embedding para rs{rs_id}...")

            embedding = get_embedding(content)

            cur.execute(
                """
                UPDATE variants
                SET embedding = %s
                WHERE id = %s
                """,
                (embedding, row_id)
            )

    conn.commit()

print("Embeddings generados correctamente.")