import re

import langextract as lx
import psycopg


DATABASE_URL = "dbname=rag_demo"
OLLAMA_URL = "http://localhost:11434"

MODEL = "qwen3:8b"


# Instrucciones para LangExtract.
PROMPT = """
Analiza la pregunta y extrae únicamente los datos que permitan
identificar variantes genéticas en una base de datos dbSNP.

Debes reconocer estas categorías:

1. rs_id:
   Un identificador de variante que comienza por rs,
   seguido de uno o más dígitos.
   Ejemplo: rs328.

2. gene:
   Un símbolo de gen mencionado explícitamente.
   Ejemplos: LPL, BRCA1, TP53.

3. chromosome:
   Un cromosoma mencionado explícitamente como tal.
   Ejemplos: cromosoma 8, chromosome 22, chr22.

Reglas:
- Extrae el texto exacto presente en la pregunta.
- No inventes identificadores, genes ni cromosomas.
- No interpretes los dígitos de un rs_id como un cromosoma.
- No extraigas números que correspondan a posiciones genómicas
  como si fueran cromosomas.
- Si no hay información que corresponda a estas categorías,
  no extraigas nada.
"""


# Ejemplos que ayudan al modelo a reconocer las categorías.
EXAMPLES = [
    lx.data.ExampleData(
        text="¿En qué cromosoma está la variante rs328?",
        extractions=[
            lx.data.Extraction(
                extraction_class="rs_id",
                extraction_text="rs328",
            )
        ],
    ),
    lx.data.ExampleData(
        text="Busca las variantes del gen LPL.",
        extractions=[
            lx.data.Extraction(
                extraction_class="gene",
                extraction_text="LPL",
            )
        ],
    ),
    lx.data.ExampleData(
        text="¿Qué variantes hay en el cromosoma 22?",
        extractions=[
            lx.data.Extraction(
                extraction_class="chromosome",
                extraction_text="22",
            )
        ],
    ),
    lx.data.ExampleData(
        text="Busca las variantes del gen BRCA1 en el cromosoma 17.",
        extractions=[
            lx.data.Extraction(
                extraction_class="gene",
                extraction_text="BRCA1",
            ),
            lx.data.Extraction(
                extraction_class="chromosome",
                extraction_text="17",
            ),
        ],
    ),
]


def extract_filters(question):
    """
    Extrae filtros estructurados de una pregunta utilizando LangExtract.

    Devuelve un diccionario con los atributos identificados.
    """

    result = lx.extract(
        text_or_documents=question,
        prompt_description=PROMPT,
        examples=EXAMPLES,
        model_id=MODEL,
        model_url=OLLAMA_URL,
    )

    filters = {}

    for extraction in result.extractions:
        category = extraction.extraction_class
        value = extraction.extraction_text.strip()

        if category == "rs_id":
            match = re.fullmatch(
                r"rs\s*0*(\d+)",
                value,
                re.IGNORECASE,
            )

            if match:
                filters["rs_id"] = int(match.group(1))

        elif category == "gene":
            # Solo aceptamos símbolos de genes con caracteres
            # alfanuméricos, guiones y puntos.
            if re.fullmatch(r"[A-Za-z0-9.-]+", value):
                filters["gene"] = value.upper()

        elif category == "chromosome":
            chromosome_match = re.fullmatch(
                r"(?:chr(?:omosome)?\s*)?(\d{1,2}|X|Y|M|MT)",
                value,
                re.IGNORECASE,
            )

            if chromosome_match:
                filters["chromosome"] = (
                    chromosome_match.group(1).upper()
                )

    return filters


def search_structured(question, k=5):
    """
    Extrae filtros con LangExtract y consulta PostgreSQL.

    Devuelve:
        filters: atributos extraídos de la pregunta.
        rows: variantes encontradas en la base de datos.
    """

    filters = extract_filters(question)

    # Si no encontramos atributos estructurados, dejamos que
    # el sistema principal decida si debe utilizar búsqueda vectorial.
    if not filters:
        return filters, []

    conditions = []
    parameters = []

    # Los nombres de las columnas están definidos en el código.
    # Los valores se pasan como parámetros, no como SQL concatenado.

    if "rs_id" in filters:
        conditions.append("rs_id = %s")
        parameters.append(filters["rs_id"])

    if "gene" in filters:
        conditions.append("gene = %s")
        parameters.append(filters["gene"])

    if "chromosome" in filters:
        conditions.append("chromosome = %s")
        parameters.append(filters["chromosome"])

    sql = f"""
        SELECT rs_id, gene, content
        FROM variants
        WHERE {" AND ".join(conditions)}
        ORDER BY rs_id
        LIMIT %s
    """

    parameters.append(k)

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, parameters)
            rows = cur.fetchall()

    return filters, rows


if __name__ == "__main__":
    question = input("Pregunta: ")

    filters, rows = search_structured(question)

    print("\nFiltros identificados:")
    print(filters)

    print("\nResultados de PostgreSQL:")

    if not rows:
        print("No se encontraron variantes.")
    else:
        for rs_id, gene, content in rows:
            print(f"\nrs{rs_id} | Gen: {gene}")
            print(content)