import bz2
import json

INPUT_FILE = "data/dbsnp/refsnp-chr22.json.bz2"
OUTPUT_FILE = "data/documents.jsonl"

MAX_VARIANTS = 10_000


def extract_variant(data):
    rs_id = int(data["refsnp_id"])

    placements = data["primary_snapshot_data"]["placements_with_allele"]

    placement = None
    assembly = None

    # Buscar placement GRCh38 de cromosoma y top-level
    for p in placements:
        traits = p["placement_annot"].get(
            "seq_id_traits_by_assembly",
            []
        )

        for trait in traits:
            if (
                trait.get("assembly_name", "").startswith("GRCh38")
                and trait.get("is_chromosome") is True
                and trait.get("is_top_level") is True
            ):
                placement = p
                assembly = trait["assembly_name"]
                break

        if placement:
            break

    # Algunos RefSNP pueden no tener este placement
    if placement is None:
        return None

    # Ejemplo:
    # NC_000008.11 -> 8
    seq_id = placement["seq_id"]

    chromosome = (
        seq_id.split("_")[1]
        .split(".")[0]
        .lstrip("0")
    )

    # SPDI utiliza posiciones 0-based.
    # Nosotros almacenamos posición 1-based.
    position = (
        placement["alleles"][0]["allele"]["spdi"]["position"]
        + 1
    )

    # Obtener alelos únicos
    alleles = []

    for allele_data in placement["alleles"]:
        inserted = allele_data["allele"]["spdi"]["inserted_sequence"]

        if inserted not in alleles:
            alleles.append(inserted)

    # Buscar genes
    genes = []

    annotations = data["primary_snapshot_data"].get(
        "allele_annotations",
        []
    )

    for annotation in annotations:
        for assembly_annotation in annotation.get(
            "assembly_annotation",
            []
        ):
            for gene in assembly_annotation.get("genes", []):
                locus = gene.get("locus")

                if locus and locus not in genes:
                    genes.append(locus)

    gene_text = ", ".join(genes) if genes else None

    # Crear documento textual para el embedding
    content = (
        f"RefSNP rs{rs_id}. "
        f"Assembly {assembly}. "
        f"Chromosome {chromosome}. "
        f"Position {position}. "
        f"Alleles: {', '.join(alleles)}. "
    )

    if gene_text:
        content += f"Associated gene: {gene_text}."

    return {
        "rs_id": rs_id,
        "assembly": assembly,
        "chromosome": chromosome,
        "position": position,
        "alleles": alleles,
        "gene": gene_text,
        "content": content,
    }


def main():
    processed = 0
    extracted = 0
    skipped = 0

    print(f"Archivo de entrada: {INPUT_FILE}")
    print(f"Objetivo: {MAX_VARIANTS:,} variantes")
    print()

    with (
        bz2.open(INPUT_FILE, "rt") as input_file,
        open(OUTPUT_FILE, "w") as output_file
    ):
        for line in input_file:

            if extracted >= MAX_VARIANTS:
                break

            line = line.strip()

            if not line:
                continue

            processed += 1

            try:
                data = json.loads(line)
                variant = extract_variant(data)

                if variant is None:
                    skipped += 1
                    continue

                output_file.write(
                    json.dumps(
                        variant,
                        ensure_ascii=False
                    ) + "\n"
                )

                extracted += 1

                if extracted % 100 == 0:
                    print(
                        f"Extraídas: "
                        f"{extracted:,}/{MAX_VARIANTS:,} "
                        f"| procesadas: {processed:,} "
                        f"| omitidas: {skipped:,}"
                    )

            except Exception as e:
                skipped += 1

                print(
                    f"Error procesando registro "
                    f"{processed}: {e}"
                )

    print()
    print("Proceso terminado.")
    print(f"Registros procesados: {processed:,}")
    print(f"Variantes extraídas:  {extracted:,}")
    print(f"Registros omitidos:   {skipped:,}")
    print(f"Archivo generado:     {OUTPUT_FILE}")


if __name__ == "__main__":
    main()