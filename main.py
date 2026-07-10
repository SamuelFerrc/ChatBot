import os

from rag_system import get_client, get_or_create_store_name

PERSONALIDADE = os.getenv(
    "CHATBOT_PERSONALITY",
    (
        "Seu nome é RAG. "
        "Você é um assistente cordial, claro e objetivo. "
        "Responda em português do Brasil e mantenha as respostas fiéis "
        "às informações recuperadas dos documentos."
    ),
)

MEMORIA_CURTA_TAMANHO = int(os.getenv("SHORT_MEMORY_TURNS", "6"))


def formatar_memoria_curta(historico):
    if not historico:
        return "Sem histórico anterior nesta conversa."

    interacoes_recentes = historico[-MEMORIA_CURTA_TAMANHO:]
    linhas = []
    for interacao in interacoes_recentes:
        linhas.append(f"Usuário: {interacao['pergunta']}")
        linhas.append(f"Assistente: {interacao['resposta']}")

    return "\n".join(linhas)


def responder(client, store_name, pergunta, historico):
    memoria_curta = formatar_memoria_curta(historico)
    interaction = client.interactions.create(
        model="gemini-3.1-pro-preview",
        system_instruction=PERSONALIDADE,
        input=(
            "Use o histórico recente apenas para manter continuidade da conversa. "
            "Para fatos sobre os documentos, priorize as informações recuperadas "
            "pelo RAG.\n\n"
            f"Histórico recente:\n{memoria_curta}\n\n"
            "Responda utilizando as informações recuperadas dos documentos. "
            "Caso os documentos não contenham a resposta, diga claramente "
            "que a informação não foi encontrada.\n\n"
            f"Pergunta: {pergunta}"
        ),
        tools=[
            {
                "type": "file_search",
                "file_search_store_names": [store_name],
                "top_k": 10,
            }
        ],
    )

    return interaction.output_text


def main():
    client = get_client()
    store_name = get_or_create_store_name(client=client)
    historico = []

    print("Chatbot RAG pronto. Digite 'sair' para encerrar.\n")

    while True:
        pergunta = input("Pergunta: ").strip()
        if pergunta.lower() in {"sair", "exit", "quit"}:
            break
        if not pergunta:
            continue

        resposta = responder(client, store_name, pergunta, historico)
        historico.append({"pergunta": pergunta, "resposta": resposta})

        print("\nResposta:")
        print(resposta)
        print()


if __name__ == "__main__":
    main()
