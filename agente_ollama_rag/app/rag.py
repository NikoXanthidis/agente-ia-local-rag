from __future__ import annotations

from pathlib import Path
from typing import Iterable
import hashlib
import re

import chromadb
from pypdf import PdfReader

from .config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    TOP_K,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MAX_SEMANTIC_DISTANCE,
    MAX_CONTEXT_CHARS,
)
from .ollama_client import embed_text, embed_texts
#Esse arquivo é responsável por gerenciar a indexação e busca de documentos PDF usando embeddings e um banco de dados vetorial. Ele fornece funções para extrair texto de PDFs, dividir o texto em trechos, gerar embeddings, armazenar os trechos no banco de dados e realizar buscas semânticas e exatas, retornando resultados relevantes para consultas do usuário.
client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)


def _clean_text(text: str) -> str:
    return " ".join((text or "").split())


def _chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    text = _clean_text(text)
    if not text:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            window_start = max(start + int(chunk_size * 0.65), start)
            candidates = [
                text.rfind(". ", window_start, end),
                text.rfind("; ", window_start, end),
                text.rfind(": ", window_start, end),
            ]
            best = max(candidates)
            if best > start:
                end = best + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(0, end - overlap)

    return chunks


def _stable_id(source: str, page: int, chunk_index: int, text: str) -> str:
    raw = f"{source}|{page}|{chunk_index}|{text}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def extract_pdf(pdf_path: Path) -> Iterable[tuple[int, str]]:
    reader = PdfReader(str(pdf_path))
    for page_number, page in enumerate(reader.pages, start=1):
        yield page_number, page.extract_text() or ""


def index_pdf(pdf_path: Path) -> dict:
    try:
        collection.delete(where={"source": pdf_path.name})
    except Exception:
        pass

    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict] = []
    empty_pages = 0

    for page_number, page_text in extract_pdf(pdf_path):
        chunks = _chunk_text(page_text)
        if not chunks:
            empty_pages += 1
            continue

        for chunk_index, chunk in enumerate(chunks):
            ids.append(_stable_id(pdf_path.name, page_number, chunk_index, chunk))
            docs.append(chunk)
            metas.append(
                {
                    "source": pdf_path.name,
                    "page": page_number,
                    "chunk": chunk_index,
                }
            )

    # Em lote: muito mais rápido que uma requisição por chunk.
    embeddings = embed_texts(docs) if docs else []

    if docs:
        collection.upsert(
            ids=ids,
            documents=docs,
            metadatas=metas,
            embeddings=embeddings,
        )

    return {
        "arquivo": pdf_path.name,
        "trechos_indexados": len(docs),
        "paginas_sem_texto": empty_pages,
    }


def extract_error_code(query: str) -> str | None:
    match = re.search(r"\bE?(\d{4,5})\b", (query or "").upper())
    return match.group(1) if match else None


def search_exact_code(code: str) -> list[dict]:
    try:
        result = collection.get(
            where_document={"$contains": code},
            include=["documents", "metadatas"],
        )
    except Exception:
        return []

    items = []
    for doc, meta in zip(result.get("documents", []), result.get("metadatas", [])):
        items.append(
            {
                "text": doc,
                "source": meta.get("source", "desconhecido"),
                "page": meta.get("page"),
                "distance": 0.0,
                "match_type": "exact",
            }
        )
    return items


def search_semantic(query: str, top_k: int = TOP_K) -> list[dict]:
    total = collection.count()
    if total == 0:
        return []

    query_embedding = embed_text(query)
    n_results = min(max(top_k + 2, top_k), total)
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    items = []
    for doc, meta, distance in zip(docs, metas, distances):
        if distance is not None and distance > MAX_SEMANTIC_DISTANCE:
            continue
        items.append(
            {
                "text": doc,
                "source": meta.get("source", "desconhecido"),
                "page": meta.get("page"),
                "distance": float(distance) if distance is not None else None,
                "match_type": "semantic",
            }
        )

    return items[:top_k]


def search(query: str, top_k: int = TOP_K) -> list[dict]:
    results: list[dict] = []

    code = extract_error_code(query)
    if code:
        results.extend(search_exact_code(code))

    results.extend(search_semantic(query, top_k=top_k))

    unique: list[dict] = []
    seen = set()

    for item in results:
        key = (item["source"], item["page"], item["text"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)

    unique.sort(
        key=lambda item: (
            0 if item.get("match_type") == "exact" else 1,
            item.get("distance") if item.get("distance") is not None else 999,
        )
    )

    return unique[:top_k]


def build_context(items: list[dict]) -> str:
    if not items:
        return ""

    blocks: list[str] = []
    used = 0

    for i, item in enumerate(items, start=1):
        text = item["text"]
        block = (
            f"[Fonte {i}]\n"
            f"Documento: {item['source']}\n"
            f"Página: {item['page']}\n"
            f"Tipo: {item.get('match_type', 'semantic')}\n"
            f"Conteúdo: {text}"
        )

        remaining = MAX_CONTEXT_CHARS - used
        if remaining <= 0:
            break
        if len(block) > remaining:
            block = block[:remaining] + "…"

        blocks.append(block)
        used += len(block) + 2

    return "\n\n".join(blocks)


def sources_from_items(items: list[dict]) -> list[dict]:
    sources = []
    seen = set()
    for item in items:
        key = (item["source"], item["page"])
        if key in seen:
            continue
        seen.add(key)
        sources.append({"arquivo": item["source"], "pagina": item["page"]})
    return sources


def stats() -> dict:
    return {"trechos": collection.count()}
