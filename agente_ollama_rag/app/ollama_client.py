from __future__ import annotations

import requests

from .config import (
    OLLAMA_URL,
    LLM_MODEL,
    EMBED_MODEL,
    NUM_CTX,
    NUM_PREDICT,
    MAX_HISTORY_ITEM_CHARS,
)
#Esse arquivo é responsável por gerenciar a comunicação com o Ollama, incluindo a geração de respostas do modelo de linguagem e a criação de embeddings para textos. Ele fornece funções para enviar solicitações HTTP ao Ollama, tratar respostas e erros, e garantir que os dados sejam processados corretamente.

def _post(path: str, payload: dict, timeout: int = 300) -> dict:
    url = f"{OLLAMA_URL}{path}"

    try:
        response = requests.post(url, json=payload, timeout=timeout)
    except requests.ConnectionError as exc:
        raise RuntimeError(
            f"Não foi possível conectar ao Ollama em {OLLAMA_URL}. "
            "Verifique se o Ollama está aberto."
        ) from exc
    except requests.Timeout as exc:
        raise RuntimeError(
            f"O Ollama demorou mais de {timeout} segundos para responder."
        ) from exc

    if not response.ok:
        try:
            detail = response.json()
        except Exception:
            detail = response.text

        extra = ""
        if response.status_code >= 500:
            extra = (
                " O Ollama falhou ao executar o modelo. Isso pode acontecer por "
                "falta de RAM/VRAM. Use LLM_MODEL='llama3.2' e feche modelos grandes."
            )

        raise RuntimeError(
            f"Erro do Ollama em {path} [HTTP {response.status_code}]: {detail}.{extra}"
        )

    return response.json()


def embed_text(text: str) -> list[float]:
    text = (text or "").strip()
    if not text:
        raise ValueError("Não é possível gerar embedding de texto vazio.")

    data = _post(
        "/api/embed",
        {"model": EMBED_MODEL, "input": text, "keep_alive": "5m"},
        timeout=180,
    )
    embeddings = data.get("embeddings") or []
    if not embeddings:
        raise RuntimeError(f"O modelo {EMBED_MODEL} não retornou embeddings.")
    return embeddings[0]


def embed_texts(texts: list[str], batch_size: int = 16) -> list[list[float]]:
    """Gera embeddings em lote para indexar PDFs com menos chamadas ao Ollama."""
    clean = [(text or "").strip() for text in texts]
    if any(not text for text in clean):
        raise ValueError("A lista de embeddings contém texto vazio.")

    all_embeddings: list[list[float]] = []

    for start in range(0, len(clean), batch_size):
        batch = clean[start : start + batch_size]
        data = _post(
            "/api/embed",
            {"model": EMBED_MODEL, "input": batch, "keep_alive": "5m"},
            timeout=300,
        )
        embeddings = data.get("embeddings") or []
        if len(embeddings) != len(batch):
            raise RuntimeError(
                f"O modelo {EMBED_MODEL} retornou {len(embeddings)} embeddings "
                f"para um lote de {len(batch)} textos."
            )
        all_embeddings.extend(embeddings)

    return all_embeddings


def _trim_history(history: list[dict] | None) -> list[dict]:
    if not history:
        return []

    result = []
    for item in history:
        role = item.get("role")
        content = " ".join((item.get("content") or "").split())
        if role not in {"user", "assistant"} or not content:
            continue
        if len(content) > MAX_HISTORY_ITEM_CHARS:
            content = content[:MAX_HISTORY_ITEM_CHARS] + "…"
        result.append({"role": role, "content": content})
    return result


def generate_answer(
    system_prompt: str,
    user_prompt: str,
    history: list[dict] | None = None,
) -> str:
    """Uma única chamada ao LLM por mensagem."""
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(_trim_history(history))
    messages.append({"role": "user", "content": user_prompt})

    data = _post(
        "/api/chat",
        {
            "model": LLM_MODEL,
            "stream": False,
            "messages": messages,
            "keep_alive": "5m",
            "options": {
                "temperature": 0.2,
                "num_ctx": NUM_CTX,
                "num_predict": NUM_PREDICT,
            },
        },
        timeout=300,
    )

    message = data.get("message") or {}
    content = (message.get("content") or "").strip()
    if not content:
        raise RuntimeError("O modelo retornou uma resposta vazia.")
    return content


def check_ollama() -> dict:
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.ConnectionError as exc:
        raise RuntimeError(f"Ollama não está acessível em {OLLAMA_URL}.") from exc
