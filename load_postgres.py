import json
import psycopg

DATABASE_URL = "dbname=rag_demo"
INPUT_FILE = "data/documents.jsonl"


def main():
    documents = []

    with open(INPUT_FILE) as f:
        for line in f:
            documents.append(json.loads(line))

    print(f"Documentos encontrados: {len(documents):,}")

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:

            cur.executemany(
                """
                INSERT INTO variants (
                    rs_id,
                    assembly,
                    chromosome,
                    position,
                    alleles,
                    gene,
                    content
                )
                VALUES (
                    %(rs_id)s,
                    %(assembly)s,
                    %(chromosome)s,
                    %(position)s,
                    %(alleles)s,
                    %(gene)s,
                    %(content)s
                )
                ON CONFLICT (rs_id) DO NOTHING
                """,
                documents
            )

        conn.commit()

    print("Documentos cargados correctamente.")


if __name__ == "__main__":
    main()