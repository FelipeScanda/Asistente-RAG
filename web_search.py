"""Búsqueda bibliográfica en PubMed mediante las E-utilities de NCBI."""

import json
import xml.etree.ElementTree as element_tree

import requests
from langchain_core.tools import tool


EUTILS_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_URL = "https://pubmed.ncbi.nlm.nih.gov"
REQUEST_TIMEOUT = 15
MAX_RESULTS = 3
MAX_ABSTRACT_CHARS = 700
HEADERS = {"User-Agent": "GenomIA-RAG/1.0"}


def get_text(element):
    """Extrae texto de un elemento XML, incluyendo etiquetas internas."""

    if element is None:
        return ""

    return " ".join("".join(element.itertext()).split())


def get_publication_date(article):
    """Obtiene la fecha de publicación disponible en el registro de PubMed."""

    date = article.find(".//ArticleDate") or article.find(".//PubDate")

    if date is None:
        return ""

    parts = [
        get_text(date.find(part))
        for part in ("Year", "Month", "Day", "MedlineDate")
    ]

    return " ".join(part for part in parts if part)


def get_authors(article):
    """Devuelve una lista breve de autores para la referencia bibliográfica."""

    authors = []

    for author in article.findall(".//AuthorList/Author")[:6]:
        collective_name = get_text(author.find("CollectiveName"))

        if collective_name:
            authors.append(collective_name)
            continue

        last_name = get_text(author.find("LastName"))
        initials = get_text(author.find("Initials"))

        if last_name:
            authors.append(f"{last_name} {initials}".strip())

    return authors


def parse_articles(xml_content, requested_ids):
    """Convierte la respuesta XML de EFetch en fuentes compactas y citables."""

    root = element_tree.fromstring(xml_content)
    sources_by_pmid = {}

    for record in root.findall(".//PubmedArticle"):
        pmid = get_text(record.find(".//PMID"))
        article = record.find(".//Article")

        if not pmid or article is None:
            continue

        abstract_parts = []

        for abstract in article.findall(".//Abstract/AbstractText"):
            label = abstract.attrib.get("Label", "")
            text = get_text(abstract)

            if text:
                abstract_parts.append(f"{label}: {text}" if label else text)

        abstract = " ".join(abstract_parts)

        if len(abstract) > MAX_ABSTRACT_CHARS:
            abstract = f"{abstract[:MAX_ABSTRACT_CHARS].rstrip()}…"

        authors = get_authors(article)
        if len(article.findall(".//AuthorList/Author")) > len(authors):
            authors.append("et al.")

        journal = get_text(article.find(".//Journal/Title"))
        publication_date = get_publication_date(article)
        title = get_text(article.find(".//ArticleTitle"))
        doi = ""

        for article_id in record.findall(".//ArticleId"):
            if article_id.attrib.get("IdType") == "doi":
                doi = get_text(article_id)
                break

        source = {
            "id": pmid,
            "title": title,
            "authors": authors,
            "journal": journal,
            "publication_date": publication_date,
            "doi": doi,
            "abstract": abstract,
            "url": f"{PUBMED_URL}/{pmid}/",
        }

        source["reference"] = (
            f"{', '.join(authors)}. {title}. {journal}. "
            f"{publication_date}. PMID: {pmid}. {source['url']}"
        ).strip()
        sources_by_pmid[pmid] = source

    return [sources_by_pmid[pmid] for pmid in requested_ids if pmid in sources_by_pmid]


def search_pubmed(query, limit=MAX_RESULTS):
    """Busca artículos de PubMed y devuelve evidencia y referencias."""

    limit = max(1, min(limit, MAX_RESULTS))

    try:
        search_response = requests.get(
            f"{EUTILS_BASE_URL}/esearch.fcgi",
            params={
                "db": "pubmed",
                "term": query,
                "retmax": limit,
                "retmode": "json",
                "sort": "relevance",
                "tool": "genomia_rag",
            },
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        search_response.raise_for_status()
        article_ids = search_response.json()["esearchresult"].get("idlist", [])

        if not article_ids:
            return {"query": query, "sources": []}

        fetch_response = requests.get(
            f"{EUTILS_BASE_URL}/efetch.fcgi",
            params={
                "db": "pubmed",
                "id": ",".join(article_ids),
                "retmode": "xml",
                "tool": "genomia_rag",
            },
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        fetch_response.raise_for_status()

        return {
            "query": query,
            "sources": parse_articles(fetch_response.content, article_ids),
        }
    except (requests.RequestException, ValueError, element_tree.ParseError) as error:
        return {"query": query, "sources": [], "error": str(error)}
