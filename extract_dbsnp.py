import json
import glob

INPUT_FILES = glob.glob("data/dbsnp/rs*.json")
OUTPUT_FILE = "data/documents.jsonl"


def extract_variant(filename):
    with open(filename) as f:
        data = json.load(f)

    rs_id = int(data["refsnp_id"])

    placements = data["primary_snapshot_data"]["placements_with_allele"]

    placement = None
    assembly = None

    for p in placements:
        traits = p["placement_annot"].get(
            "seq_id_traits_by_assembly", []
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

    if placement is None:
        return None

    seq_id = placement["seq_id"]

    chromosome = seq_id.split("_")[1].split(".")[0].lstrip("0")

    position = (
        placement["alleles"][0]["allele"]["spdi"]["position"] + 1
    )

    alleles = []

    for allele_data in placement["alleles"]:
        inserted = allele_data["allele"]["spdi"]["inserted_sequence"]

        if inserted not in alleles:
            alleles.append(inserted)

    genes = []

    annotations = data["primary_snapshot_data"].get(
        "allele_annotations", []
    )

    for annotation in annotations:
        for assembly_annotation in annotation.get(
            "assembly_annotation", []
        ):
            for gene in assembly_annotation.get("genes", []):
                locus = gene.get("locus")

                if locus and locus not in genes:
                    genes.append(locus)

    gene_text = ", ".join(genes) if genes else None

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


with open(OUTPUT_FILE, "w") as output:

    for filename in INPUT_FILES:
        print(f"Procesando {filename}...")

        variant = extract_variant(filename)

        if variant:
            output.write(json.dumps(variant) + "\n")

            print(json.dumps(variant, indent=2))
        else:
            print("No se encontró un placement GRCh38.")

print(f"\nDocumentos guardados en: {OUTPUT_FILE}")