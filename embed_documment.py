import requests
import psycopg
import time

DATABASE_URL = "dbname=rag_demo"

OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL = "nomic-embed-text"


def get_embedding(text):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": text
        },
        timeout=120
    )

    response.raise_for_status()

    return response.json()["embedding"]


def main():

    with psycopg.connect(DATABASE_URL) as conn:

        with conn.cursor() as cur:

            cur.execute("""
                SELECT id, rs_id, content
                FROM variants
                WHERE embedding IS NULL
                ORDER BY id
            """)

            rows = cur.fetchall()

            total = len(rows)

            print(f"Documentos pendientes: {total:,}")
            print()

            if total == 0:
                print("No hay documentos pendientes.")
                return

            completed = 0
            errors = 0

            start_time = time.time()

            for row_id, rs_id, content in rows:

                completed += 1

                print(
                    f"[{completed:,}/{total:,}] "
                    f"Generando embedding para rs{rs_id}...",
                    end=" ",
                    flush=True
                )

                try:

                    embedding = get_embedding(content)

                    cur.execute(
                        """
                        UPDATE variants
                        SET embedding = %s
                        WHERE id = %s
                        """,
                        (embedding, row_id)
                    )

                    # Guardar inmediatamente
                    conn.commit()

                    elapsed = time.time() - start_time

                    average = elapsed / completed
                    remaining = average * (total - completed)

                    print(
                        f"✓ "
                        f"(faltan aprox. "
                        f"{remaining / 60:.1f} min)"
                    )

                except Exception as e:

                    errors += 1

                    print(f"✗ Error: {e}")

            print()
            print("--------------------------------")
            print("Proceso terminado")
            print("--------------------------------")
            print(f"Procesados: {completed:,}")
            print(f"Errores:    {errors:,}")


if __name__ == "__main__":
    main()