import json

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from embedding import search as search_semantic
from search import search_structured
from web_search import search_pubmed


OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "qwen3:8b"


SYSTEM_PROMPT = """
Eres Genomin, un asistente especializado en genómica. No respondas preguntas
factuales, conceptuales ni científicas usando solamente tu conocimiento interno:
antes debes llamar exactamente una herramienta de búsqueda. Si dudas entre la
búsqueda semántica y buscar_en_internet, prioriza buscar_en_internet.

Selecciona la herramienta siguiendo este orden:
1. buscar_estructuradamente: cuando se mencione explícitamente un rs_id, un gen
   o un cromosoma y la respuesta pueda obtenerse de la base local.
2. buscar_semanticamente: solo para preguntas sobre variantes presentes en la
   base local que no tengan filtros exactos.
3. buscar_en_internet: para conceptos, mecanismos biológicos, evidencia
   científica, información reciente, preguntas que pidan bibliografía o toda
   pregunta cuya respuesta no esté claramente limitada a la base local.

Las preguntas que soliciten bibliografía, artículos, evidencia o fuentes deben
usar siempre buscar_en_internet. Nunca inventes, completes ni cites referencias
que no hayan sido entregadas por una herramienta.

Después de recibir el resultado de una herramienta, responde en español usando
solamente la información recuperada. Si no hay resultados o no son suficientes,
dilo claramente.

Cuando uses buscar_en_internet, cita las fuentes utilizadas en el texto como
[1], [2], etcétera, y termina con una sección "Referencias" que enumere solo
las referencias entregadas por la herramienta, incluyendo sus enlaces PubMed.

Limitate a responder preguntas sola y exclusivamente sobre genomica, genetica u
otras áreas relacionadas. Si se te pide otra cosa fuera de tu dominio, indica
al usuario que no estás capacitado para eso y no le des la respuesta. Evita a
toda costa dar código o hacer alguna otra tarea que no se encuentre en tu dominio,
solo responde amablemente que no puedes hacerlo.

Para conversación casual, saludos o peticiones que no requieran la base de datos,
no llames herramientas y responde normalmente en español.
""".strip()


def format_semantic_results(rows):
    """Convierte las filas vectoriales en datos serializables para el modelo."""

    return [
        {
            "rs_id": rs_id,
            "gene": gene,
            "content": content,
            "distance": distance,
        }
        for rs_id, gene, content, distance in rows
    ]


def format_structured_results(filters, rows):
    """Convierte las filas de PostgreSQL en datos serializables para el modelo."""

    return {
        "filters": filters,
        "results": [
            {"rs_id": rs_id, "gene": gene, "content": content}
            for rs_id, gene, content in rows
        ],
    }


@tool
def buscar_semanticamente(consulta: str, limite: int = 5) -> str:
    """Busca variantes por similitud semántica.

    Úsala para preguntas genómicas conceptuales que no identifican de forma
    explícita una variante, un gen o un cromosoma.
    """

    rows = search_semantic(consulta, k=limite)
    return json.dumps(format_semantic_results(rows), ensure_ascii=False)


@tool
def buscar_estructuradamente(consulta: str, limite: int = 5) -> str:
    """Busca variantes usando filtros exactos de rs_id, gen o cromosoma.

    Úsala cuando la pregunta mencione explícitamente alguno de esos datos.
    """

    filters, rows = search_structured(consulta, k=limite)
    return json.dumps(format_structured_results(filters, rows), ensure_ascii=False)

@tool
def buscar_en_internet(consulta: str, limite: int = 3) -> str:
    """Busca literatura biomédica en Internet mediante PubMed.

    Úsala para preguntas conceptuales, mecanismos biológicos, evidencia
    científica, información reciente o cuando el usuario solicite bibliografía.
    No la uses para buscar una variante concreta guardada en la base local.
    """

    return json.dumps(search_pubmed(consulta, limit=limite), ensure_ascii=False)

TOOLS = [buscar_semanticamente, buscar_estructuradamente, buscar_en_internet]
TOOLS_BY_NAME = {tool.name: tool for tool in TOOLS}

LLM = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0, reasoning=False)
LLM_WITH_TOOLS = LLM.bind_tools(TOOLS)


def answer_question(query, messages=None):
    """Procesa un turno y conserva el historial recibido de la conversación."""

    if messages is None:
        messages = [SystemMessage(content=SYSTEM_PROMPT)]

    messages.append(HumanMessage(content=query))

    assistant_message = LLM_WITH_TOOLS.invoke(messages)
    messages.append(assistant_message)
    tool_calls = assistant_message.tool_calls

    if not tool_calls:
        return None, None, assistant_message.content

    # El prompt solicita una sola herramienta. Rechazamos más de una para no
    # mezclar resultados de estrategias distintas de forma silenciosa.
    if len(tool_calls) != 1:
        raise RuntimeError("El modelo solicitó más de una herramienta de búsqueda.")

    tool_call = tool_calls[0]
    tool_name = tool_call["name"]
    tool_result = TOOLS_BY_NAME[tool_name].invoke(tool_call["args"])
    tool_message = ToolMessage(
        content=tool_result,
        tool_call_id=tool_call["id"],
        name=tool_name,
    )
    messages.append(tool_message)
    result = json.loads(tool_message.content)

    final_message = LLM.invoke(messages)
    messages.append(final_message)

    return tool_name, result, final_message.content


def main():
    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    print("Hola 😊 soy Genomin, el asistente inteligente de GenomIA. Puedes hacerme preguntas sobre genomica y yo las responderé. Escribe 'salir' para terminar.")

    while True:
        query = input("\nTú: ").strip()

        if query.lower() in {"salir", "exit", "quit"}:
            print("Hasta luego.")
            break

        if not query:
            continue

        tool_name, results, answer = answer_question(query, messages)

        if tool_name:
            print(f"\nHerramienta elegida: {tool_name}")
            print("Resultados recuperados:")
            print("-----------------------")
            print(json.dumps(results, ensure_ascii=False, indent=2))

        print("\nAsistente:")
        print("-----------")
        print(answer)


if __name__ == "__main__":
    main()
