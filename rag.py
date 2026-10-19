import json

import requests

from embedding import search as search_semantic
from search import search_structured


OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
LLM_MODEL = "qwen3:8b"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "buscar_semanticamente",
            "description": (
                "Busca variantes por similitud semántica. Úsala para preguntas "
                "conceptuales o de lenguaje natural que no identifican de forma "
                "explícita una variante, gen o cromosoma."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "consulta": {
                        "type": "string",
                        "description": "La pregunta del usuario.",
                    },
                    "limite": {
                        "type": "integer",
                        "description": "Máximo de resultados a recuperar.",
                    },
                },
                "required": ["consulta"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "buscar_estructuradamente",
            "description": (
                "Busca variantes mediante filtros exactos de rs_id, gen o "
                "cromosoma. Úsala cuando la pregunta mencione explícitamente "
                "alguno de esos datos."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "consulta": {
                        "type": "string",
                        "description": "La pregunta del usuario.",
                    },
                    "limite": {
                        "type": "integer",
                        "description": "Máximo de resultados a recuperar.",
                    },
                },
                "required": ["consulta"],
            },
        },
    },
]


SYSTEM_PROMPT = """
Eres un asistente especializado en genómica. Para responder la pregunta del
usuario debes llamar exactamente una de las herramientas de búsqueda disponibles
antes de contestar. Elige la búsqueda estructurada si se menciona explícitamente
un rs_id, un gen o un cromosoma; en los demás casos, elige la búsqueda semántica.

Después de recibir el resultado de la herramienta, responde en español usando
solamente la información recuperada. Si no hay resultados o no bastan para
responder, dilo claramente. No inventes información.
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


def run_tool(tool_call):
    """Ejecuta una de las herramientas que el modelo solicitó."""

    function = tool_call["function"]
    name = function["name"]
    arguments = function.get("arguments", {})

    # Ollama normalmente entrega un diccionario, pero aceptamos JSON para que
    # el programa siga funcionando con versiones que lo devuelvan como texto.
    if isinstance(arguments, str):
        arguments = json.loads(arguments)

    query = arguments["consulta"]
    limit = arguments.get("limite", 5)

    if name == "buscar_semanticamente":
        return format_semantic_results(search_semantic(query, k=limit))

    if name == "buscar_estructuradamente":
        filters, rows = search_structured(query, k=limit)
        return format_structured_results(filters, rows)

    raise ValueError(f"Herramienta no reconocida: {name}")


def answer_question(query):
    """Deja que el modelo elija y utilice una estrategia de recuperación."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    response = requests.post(
        OLLAMA_CHAT_URL,
        json={"model": LLM_MODEL, "messages": messages, "tools": TOOLS, "stream": False},
    )
    response.raise_for_status()

    assistant_message = response.json()["message"]
    messages.append(assistant_message)
    tool_calls = assistant_message.get("tool_calls", [])

    if not tool_calls:
        raise RuntimeError("El modelo no seleccionó una herramienta de búsqueda.")

    # El prompt solicita una sola herramienta. Rechazamos más de una para no
    # mezclar resultados de estrategias distintas de forma silenciosa.
    if len(tool_calls) != 1:
        raise RuntimeError("El modelo solicitó más de una herramienta de búsqueda.")

    tool_call = tool_calls[0]
    result = run_tool(tool_call)
    tool_name = tool_call["function"]["name"]

    messages.append(
        {
            "role": "tool",
            "tool_name": tool_name,
            "content": json.dumps(result, ensure_ascii=False),
        }
    )

    response = requests.post(
        OLLAMA_CHAT_URL,
        json={"model": LLM_MODEL, "messages": messages, "stream": False},
    )
    response.raise_for_status()

    return tool_name, result, response.json()["message"]["content"]


def main():
    query = input("Pregunta: ")
    tool_name, results, answer = answer_question(query)

    print(f"\nHerramienta elegida: {tool_name}")
    print("Resultados recuperados:")
    print("-----------------------")
    print(json.dumps(results, ensure_ascii=False, indent=2))

    print("\nRespuesta:")
    print("---------")
    print(answer)


if __name__ == "__main__":
    main()
