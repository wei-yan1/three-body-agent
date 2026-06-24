"""Build structure-aware RAG chunks and Chroma indexes."""

from __future__ import annotations

import argparse
import datetime
import json
import time
import urllib.request
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.embeddings.structure_aware_chunker import (
    DEFAULT_RAG_CHUNK_DIR,
    build_and_write_default_chunks,
    build_persona_structure_documents,
)


DEFAULT_CHROMA_DIR = Path("data/indexes/chroma")
DEFAULT_NOVEL_CHUNKS_PATH = DEFAULT_RAG_CHUNK_DIR / "novel_chunks.jsonl"
DEFAULT_ASYNC_INPUT_PATH = DEFAULT_RAG_CHUNK_DIR / "novel_embedding_async_input.txt"
DEFAULT_ASYNC_RESULT_PATH = DEFAULT_RAG_CHUNK_DIR / "novel_embedding_async_result.jsonl"
PERSONA_COLLECTION = "three_body_persona_profiles"
NOVEL_COLLECTION = "three_body_novel_chunks"


def _batched(items: list[object], batch_size: int) -> list[list[object]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def _clean_metadata_value(value: object) -> str | int | float | bool:
    if value is None:
        return ""
    if isinstance(value, str | int | float | bool):
        return value
    return json.dumps(value, ensure_ascii=False)


def _clean_document_metadata(document: object) -> object:
    document.metadata = {
        key: _clean_metadata_value(value) for key, value in document.metadata.items()
    }
    return document


def _clean_documents(documents: list[object]) -> list[object]:
    return [_clean_document_metadata(document) for document in documents]


def load_documents_from_chunk_jsonl(path: Path) -> list[object]:
    """Load auditable chunk JSONL records into LangChain documents."""
    from langchain_core.documents import Document

    documents: list[object] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = record.get("text")
            metadata = record.get("metadata")
            if not isinstance(text, str) or not isinstance(metadata, dict):
                message = f"Invalid chunk record at {path}:{line_number}"
                raise ValueError(message)
            documents.append(Document(page_content=text, metadata=metadata))
    return documents


def _reset_chroma_collection(persist_directory: Path, collection_name: str) -> None:
    """Delete one Chroma collection before rebuilding it from JSONL."""
    import chromadb

    persist_directory.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_directory))
    try:
        client.delete_collection(collection_name)
    except Exception:
        # Chroma raises different exception types across versions when a
        # collection does not exist. Missing collection is the desired state.
        pass


def build_chunks(output_dir: Path = DEFAULT_RAG_CHUNK_DIR) -> dict[str, int]:
    """Build auditable JSONL chunk files."""
    return build_and_write_default_chunks(output_dir)


def build_chroma_indexes(
    persist_directory: Path = DEFAULT_CHROMA_DIR,
    batch_size: int = 64,
    novel_chunks_path: Path = DEFAULT_NOVEL_CHUNKS_PATH,
) -> dict[str, int | str]:
    """Build Chroma collections using DashScope/Bailian embeddings."""
    from dotenv import load_dotenv
    from langchain_chroma import Chroma

    from app.rag.embeddings.embedding_model import create_dashscope_embeddings

    load_dotenv()
    embeddings = create_dashscope_embeddings()
    persist_directory.mkdir(parents=True, exist_ok=True)

    persona_docs = _clean_documents(build_persona_structure_documents())
    novel_docs = _clean_documents(load_documents_from_chunk_jsonl(novel_chunks_path))

    _reset_chroma_collection(persist_directory, PERSONA_COLLECTION)
    _reset_chroma_collection(persist_directory, NOVEL_COLLECTION)

    persona_store = Chroma(
        collection_name=PERSONA_COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )
    novel_store = Chroma(
        collection_name=NOVEL_COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )

    for batch in _batched(persona_docs, batch_size):
        persona_store.add_documents(
            batch,
            ids=[document.metadata["chunk_id"] for document in batch],
        )
    for batch in _batched(novel_docs, batch_size):
        novel_store.add_documents(
            batch,
            ids=[document.metadata["chunk_id"] for document in batch],
        )

    return {
        "persist_directory": str(persist_directory),
        "persona_collection": PERSONA_COLLECTION,
        "persona_documents": len(persona_docs),
        "novel_collection": NOVEL_COLLECTION,
        "novel_source": str(novel_chunks_path),
        "novel_documents": len(novel_docs),
    }


def build_novel_chroma_index(
    persist_directory: Path = DEFAULT_CHROMA_DIR,
    novel_chunks_path: Path = DEFAULT_NOVEL_CHUNKS_PATH,
    batch_size: int = 64,
) -> dict[str, int | str]:
    """Rebuild only the novel Chroma collection from novel_chunks.jsonl."""
    from dotenv import load_dotenv
    from langchain_chroma import Chroma

    from app.rag.embeddings.embedding_model import create_dashscope_embeddings

    load_dotenv()
    embeddings = create_dashscope_embeddings()
    persist_directory.mkdir(parents=True, exist_ok=True)
    novel_docs = _clean_documents(load_documents_from_chunk_jsonl(novel_chunks_path))
    _reset_chroma_collection(persist_directory, NOVEL_COLLECTION)

    novel_store = Chroma(
        collection_name=NOVEL_COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )
    for batch in _batched(novel_docs, batch_size):
        novel_store.add_documents(
            batch,
            ids=[document.metadata["chunk_id"] for document in batch],
        )

    return {
        "persist_directory": str(persist_directory),
        "novel_collection": NOVEL_COLLECTION,
        "novel_source": str(novel_chunks_path),
        "embedding_model": getattr(embeddings, "model", "unknown"),
        "novel_documents": len(novel_docs),
    }


def write_novel_async_embedding_input(
    novel_chunks_path: Path = DEFAULT_NOVEL_CHUNKS_PATH,
    output_path: Path = DEFAULT_ASYNC_INPUT_PATH,
) -> dict[str, int | str]:
    """Write async-v2 input text file from audited novel chunk JSONL.

    DashScope BatchTextEmbedding expects a URL to a plain text file containing
    one text per line. The line order must be preserved because the result is
    joined back to novel_chunks.jsonl by index.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with novel_chunks_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as target:
        for raw_line in source:
            if not raw_line.strip():
                continue
            record = json.loads(raw_line)
            text = str(record["text"]).replace("\r", " ").replace("\n", " ")
            target.write(text + "\n")
            count += 1
    return {
        "novel_source": str(novel_chunks_path),
        "async_input": str(output_path),
        "lines": count,
        "next_step": "Upload async_input to OSS or another URL reachable by DashScope, then run novel-async-submit --input-url <url>.",
    }


def submit_novel_async_embedding_task(
    input_url: str,
    model: str | None = None,
    wait: bool = False,
    poll_seconds: int = 10,
    timeout_seconds: int = 1800,
) -> dict[str, object]:
    """Submit a DashScope async text embedding task for the novel input URL."""
    from dotenv import load_dotenv

    load_dotenv()
    import os
    import dashscope

    from app.rag.embeddings.embedding_model import get_dashscope_document_embedding_model

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is required for DashScope async embedding.")
    model = model or get_dashscope_document_embedding_model()

    response = dashscope.BatchTextEmbedding.async_call(
        model=model,
        url=input_url,
        api_key=api_key,
        text_type="document",
    )
    if response.status_code != 200:
        message = f"DashScope async embedding failed: code={response.code}, message={response.message}"
        raise RuntimeError(message)

    if not wait:
        return {
            "model": model,
            "task_id": response.output.task_id,
            "task_status": response.output.task_status,
            "result_url": response.output.url,
            "next_step": "When task_status is SUCCEEDED, download result_url and run novel-async-import.",
        }

    deadline = time.time() + timeout_seconds
    current = response
    while time.time() < deadline:
        current = dashscope.BatchTextEmbedding.fetch(
            response.output.task_id,
            api_key=api_key,
        )
        if current.status_code != 200:
            message = f"DashScope async fetch failed: code={current.code}, message={current.message}"
            raise RuntimeError(message)
        if current.output.task_status in {"SUCCEEDED", "FAILED", "CANCELED"}:
            break
        time.sleep(poll_seconds)

    result = {
        "model": model,
        "task_id": current.output.task_id,
        "task_status": current.output.task_status,
        "result_url": current.output.url,
    }
    if current.output.task_status != "SUCCEEDED":
        result["next_step"] = "Task is not complete yet. Re-run with --wait or fetch later."
    return result


def download_novel_async_embedding_result(
    result_url: str,
    output_path: Path = DEFAULT_ASYNC_RESULT_PATH,
) -> dict[str, str]:
    """Download DashScope async embedding result JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(result_url, output_path)
    return {"result_url": result_url, "async_result": str(output_path)}


def upload_async_input_to_oss(
    input_path: Path = DEFAULT_ASYNC_INPUT_PATH,
    object_key: str | None = None,
    expires_hours: int = 24,
) -> dict[str, str]:
    """Upload async input file to OSS and return a presigned GET URL."""
    from dotenv import load_dotenv

    load_dotenv()
    import os
    import alibabacloud_oss_v2 as oss

    access_key_id = os.getenv("OSS_ACCESS_KEY_ID")
    access_key_secret = os.getenv("OSS_ACCESS_KEY_SECRET")
    bucket = os.getenv("OSS_BUCKET")
    endpoint = os.getenv("OSS_ENDPOINT")
    region = os.getenv("OSS_REGION")
    if not all([access_key_id, access_key_secret, bucket, endpoint, region]):
        message = (
            "OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET, OSS_BUCKET, OSS_ENDPOINT, "
            "and OSS_REGION are required in .env to upload async input."
        )
        raise RuntimeError(message)

    object_key = object_key or f"rag/three_body/{input_path.name}"
    credentials_provider = oss.credentials.StaticCredentialsProvider(
        access_key_id,
        access_key_secret,
    )
    config = oss.Config(
        region=region,
        endpoint=endpoint,
        credentials_provider=credentials_provider,
    )
    client = oss.Client(config)

    with input_path.open("rb") as file:
        client.put_object(
            oss.PutObjectRequest(
                bucket=bucket,
                key=object_key,
                body=file,
                content_type="text/plain; charset=utf-8",
            )
        )

    presigned = client.presign(
        oss.GetObjectRequest(bucket=bucket, key=object_key),
        expires=datetime.timedelta(hours=expires_hours),
    )
    return {
        "input_path": str(input_path),
        "bucket": bucket,
        "object_key": object_key,
        "input_url": presigned.url,
        "expires_hours": str(expires_hours),
        "next_step": "Run novel-async-submit --input-url <input_url> --wait.",
    }


def _extract_embedding_from_async_record(record: dict[str, object]) -> list[float]:
    if isinstance(record.get("embedding"), list):
        return record["embedding"]  # type: ignore[return-value]
    output = record.get("output")
    if isinstance(output, dict) and isinstance(output.get("embedding"), list):
        return output["embedding"]  # type: ignore[return-value]
    embeddings = record.get("embeddings")
    if isinstance(embeddings, list) and embeddings:
        first = embeddings[0]
        if isinstance(first, dict) and isinstance(first.get("embedding"), list):
            return first["embedding"]  # type: ignore[return-value]
        if isinstance(first, list):
            return first  # type: ignore[return-value]
    message = "Cannot find embedding vector in async result record."
    raise ValueError(message)


def load_async_embeddings(path: Path) -> list[list[float]]:
    """Load embedding vectors from DashScope async result JSONL."""
    embeddings: list[list[float]] = []
    with path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            if not raw_line.strip():
                continue
            record = json.loads(raw_line)
            embeddings.append(_extract_embedding_from_async_record(record))
    return embeddings


def import_novel_async_embeddings_to_chroma(
    persist_directory: Path = DEFAULT_CHROMA_DIR,
    novel_chunks_path: Path = DEFAULT_NOVEL_CHUNKS_PATH,
    async_result_path: Path = DEFAULT_ASYNC_RESULT_PATH,
    batch_size: int = 64,
) -> dict[str, int | str]:
    """Import async-v2 embeddings into Chroma without re-embedding locally."""
    from langchain_chroma import Chroma

    novel_docs = _clean_documents(load_documents_from_chunk_jsonl(novel_chunks_path))
    embeddings = load_async_embeddings(async_result_path)
    if len(novel_docs) != len(embeddings):
        message = (
            f"Novel chunk count ({len(novel_docs)}) does not match embedding count "
            f"({len(embeddings)}). Ensure async input was generated from the same JSONL."
        )
        raise ValueError(message)

    _reset_chroma_collection(persist_directory, NOVEL_COLLECTION)
    store = Chroma(
        collection_name=NOVEL_COLLECTION,
        embedding_function=None,
        persist_directory=str(persist_directory),
    )
    collection = store._collection

    for batch_start in range(0, len(novel_docs), batch_size):
        batch_docs = novel_docs[batch_start : batch_start + batch_size]
        batch_embeddings = embeddings[batch_start : batch_start + batch_size]
        collection.upsert(
            ids=[document.metadata["chunk_id"] for document in batch_docs],
            documents=[document.page_content for document in batch_docs],
            metadatas=[document.metadata for document in batch_docs],
            embeddings=batch_embeddings,
        )

    return {
        "persist_directory": str(persist_directory),
        "novel_collection": NOVEL_COLLECTION,
        "novel_source": str(novel_chunks_path),
        "async_result": str(async_result_path),
        "embedding_model": "text-embedding-async-v2",
        "novel_documents": len(novel_docs),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "chunks",
            "chroma",
            "novel-chroma",
            "novel-async-input",
            "novel-async-upload",
            "novel-async-submit",
            "novel-async-download",
            "novel-async-import",
            "all",
        ],
        help="Build JSONL chunks, Chroma indexes, or both.",
    )
    parser.add_argument(
        "--chunk-dir",
        default=str(DEFAULT_RAG_CHUNK_DIR),
        help="Directory for generated chunk JSONL files.",
    )
    parser.add_argument(
        "--chroma-dir",
        default=str(DEFAULT_CHROMA_DIR),
        help="Chroma persist directory.",
    )
    parser.add_argument(
        "--novel-chunks",
        default=str(DEFAULT_NOVEL_CHUNKS_PATH),
        help="Path to novel chunk JSONL used to rebuild the novel vector collection.",
    )
    parser.add_argument(
        "--async-input",
        default=str(DEFAULT_ASYNC_INPUT_PATH),
        help="Local async-v2 input text file generated from novel chunk JSONL.",
    )
    parser.add_argument(
        "--async-result",
        default=str(DEFAULT_ASYNC_RESULT_PATH),
        help="Local async-v2 embedding result JSONL downloaded from DashScope.",
    )
    parser.add_argument(
        "--input-url",
        help="OSS or HTTP URL for the async-v2 input text file.",
    )
    parser.add_argument(
        "--oss-key",
        help="OSS object key for uploading the async input file.",
    )
    parser.add_argument(
        "--url-expires-hours",
        type=int,
        default=24,
        help="Expiration in hours for the presigned async input URL.",
    )
    parser.add_argument(
        "--result-url",
        help="DashScope async-v2 result URL to download.",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for DashScope async task to finish after submission.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Documents per Chroma add batch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result: dict[str, object] = {}

    if args.command in {"chunks", "all"}:
        result["chunks"] = build_chunks(Path(args.chunk_dir))
    if args.command in {"chroma", "all"}:
        result["chroma"] = build_chroma_indexes(
            Path(args.chroma_dir),
            batch_size=args.batch_size,
            novel_chunks_path=Path(args.novel_chunks),
        )
    if args.command == "novel-chroma":
        result["novel_chroma"] = build_novel_chroma_index(
            persist_directory=Path(args.chroma_dir),
            novel_chunks_path=Path(args.novel_chunks),
            batch_size=args.batch_size,
        )
    if args.command == "novel-async-input":
        result["novel_async_input"] = write_novel_async_embedding_input(
            novel_chunks_path=Path(args.novel_chunks),
            output_path=Path(args.async_input),
        )
    if args.command == "novel-async-upload":
        result["novel_async_upload"] = upload_async_input_to_oss(
            input_path=Path(args.async_input),
            object_key=args.oss_key,
            expires_hours=args.url_expires_hours,
        )
    if args.command == "novel-async-submit":
        if not args.input_url:
            raise ValueError("novel-async-submit requires --input-url.")
        result["novel_async_submit"] = submit_novel_async_embedding_task(
            input_url=args.input_url,
            wait=args.wait,
        )
    if args.command == "novel-async-download":
        if not args.result_url:
            raise ValueError("novel-async-download requires --result-url.")
        result["novel_async_download"] = download_novel_async_embedding_result(
            result_url=args.result_url,
            output_path=Path(args.async_result),
        )
    if args.command == "novel-async-import":
        result["novel_async_import"] = import_novel_async_embeddings_to_chroma(
            persist_directory=Path(args.chroma_dir),
            novel_chunks_path=Path(args.novel_chunks),
            async_result_path=Path(args.async_result),
            batch_size=args.batch_size,
        )

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
