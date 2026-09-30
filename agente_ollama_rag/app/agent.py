from __future__ import annotations

import re

from .config import HISTORY_MESSAGES
from .conversation import add_message, ensure_conversation, get_messages
from .ollama_client import generate_answer
from .rag import build_context, search, sources_from_items


SYSTEM_PROMPT = """
Você é um assistente local especialista em TI, redes, Windows, Linux e diagnóstico técnico.

CONVERSA:
- Use o histórico recente para manter continuidade.
- Entenda referências como "ele", "isso", "esse erro", "como?", "e a porta?".
- Não repita explicações sem necessidade.
- Se faltar uma informação importante para diagnosticar, faça uma pergunta curta e útil.
- Responda em português do Brasil.

BASE DE CONHECIMENTO:
- Quando houver contexto recuperado relevante, priorize-o.
- Preserve exatamente códigos, comandos, IPs, portas, números e unidades.
- Se o usuário pedir explicitamente informação "do PDF", "do manual" ou "da documentação", não invente o que não estiver no contexto.
- Em perguntas gerais de TI, você pode usar seu conhecimento geral quando a base não trouxer a resposta.

DIAGNÓSTICO:
- Prefira primeiro comandos de consulta/inspeção.
- Antes de sugerir alteração de configuração, explique brevemente o efeito.
""".strip()


CASUAL = {
    "oi", "olá", "ola", "bom dia", "boa tarde", "boa noite",
    "obrigado", "obrigada", "valeu", "entendi", "ok", "certo", "beleza",
}

CONTEXTUAL_MARKERS = re.compile(
    r"\b(ele|ela|isso|isto|esse|essa|aquele|aquela|dele|dela|nisso|nele|nela|"
    r"como|por que|porque|e qual|e como|e a|e o|também|tambem)\b",
    re.IGNORECASE,
)


def _is_casual_message(text: str) -> bool:
    return " ".join(text.lower().strip().split()) in CASUAL


def _explicit_document_request(text: str) -> bool:
    return bool(
        re.search(
            r"\b(pdf|manual|documento|documentação|documentacao|base de conhecimento|arquivo)\b",
            text.lower(),
        )
    )


def _retrieval_query(question: str, history: list[dict]) -> str:
    """Contextualiza a busca sem chamar o LLM uma segunda vez."""
    question = question.strip()
    if not history:
        return question

    looks_contextual = len(question) < 120 or bool(CONTEXTUAL_MARKERS.search(question))
    if not looks_contextual:
        return question

    # As últimas mensagens ajudam o embedding a resolver "ele", "isso", etc.
    recent_parts: list[str] = []
    for item in history[-4:]:
        content = " ".join((item.get("content") or "").split())
        if not content:
            continue
        # O texto do assistente pode ser enorme, ai limitei para não estourar o embedding.
        
        limit = 420 if item.get("role") == "assistant" else 700
        recent_parts.append(content[:limit])

    return "\n".join(recent_parts + [question])[-2200:]


def answer_question(question: str, conversation_id: str | None = None) -> dict:
    question = (question or "").strip()
    if not question:
        raise ValueError("Pergunta vazia.")

    conversation_id = ensure_conversation(conversation_id)
    history = get_messages(conversation_id, limit=HISTORY_MESSAGES)

    retrieval_query = _retrieval_query(question, history)

    contexts = []
    if not _is_casual_message(question):
        contexts = search(retrieval_query)

    context_text = build_context(contexts)
    document_only = _explicit_document_request(question)

    if context_text:
        rag_block = f"CONTEXTO DA BASE:\n{context_text}"
    else:
        rag_block = "CONTEXTO DA BASE: nenhum trecho relevante foi recuperado."

    instruction = (
        "A resposta deve se limitar aos documentos recuperados. Se eles não trouxerem a informação, diga isso."
        if document_only
        else "Use a base quando for relevante; fora disso, você pode responder com conhecimento geral de TI."
    )

    user_prompt = f"""
PERGUNTA ATUAL:
{question}

{rag_block}

INSTRUÇÃO:
{instruction}
""".strip()

    # Só UMA chamada ao LLM.
    answer = generate_answer(SYSTEM_PROMPT, user_prompt, history=history)

    add_message(conversation_id, "user", question)
    add_message(conversation_id, "assistant", answer)

    return {
        "answer": answer,
        "sources": sources_from_items(contexts),
        "conversation_id": conversation_id,
        "retrieval_query": retrieval_query,
    }
