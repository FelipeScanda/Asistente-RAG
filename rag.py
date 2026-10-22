import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from embedding import search as search_semantic
from search import search_structured


OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "qwen3:8b"


SYSTEM_PROMPT = """
Eres un asistente especializado en genómica. Usa una herramienta de búsqueda
solo si la pregunta requiere información sobre variantes genéticas de la base de
datos. Elige la búsqueda estructurada si se menciona explícitamente un rs_id, un
gen o un cromosoma; para las demás preguntas genómicas sobre variantes, elige la
búsqueda semántica.

Después de recibir el resultado de la herramienta, responde en español usando
solamente la información recuperada. Si no hay resultados o no bastan para
responder, dilo claramente. No inventes información.

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


TOOLS = [buscar_semanticamente, buscar_estructuradamente]
TOOLS_BY_NAME = {tool.name: tool for tool in TOOLS}

LLM = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
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
    tool_message = TOOLS_BY_NAME[tool_name].invoke(tool_call)
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
