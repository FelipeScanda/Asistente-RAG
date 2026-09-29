import json
import psycopg

DATABASE_URL = "dbname=rag_demo"

with psycopg.connect(DATABASE_URL) as conn:

    with conn.cursor() as cur:

        with open("data/documents.jsonl") as f:

            for line in f:
                document = json.loads(line)

                cur.execute(
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
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (rs_id) DO NOTHING
                    """,
                    (
                        document["rs_id"],
                        document["assembly"],
                        document["chromosome"],
                        document["position"],
                        document["alleles"],
                        document["gene"],
                        document["content"],
                    )
                )

    conn.commit()

print("Documentos cargados correctamente.")