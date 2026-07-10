import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documentos"
STORE_CACHE_PATH = BASE_DIR / ".rag_store.json"
STORE_DISPLAY_NAME = "documentos-pesquisa"
EMBEDDING_MODEL = "models/gemini-embedding-2"


def get_client():
    return genai.Client()


def _normalize_path(path):
    path = Path(path)
    if not path.is_absolute():
        path = BASE_DIR / path

    if not path.exists():
        raise FileNotFoundError(f"Documento não encontrado: {path}")

    return path


def _configured_document_paths():
    configured_paths = os.getenv("RAG_DOCUMENT_PATHS") or os.getenv("RAG_DOCUMENT_PATH")
    if not configured_paths:
        return None

    return [
        path.strip()
        for path in configured_paths.replace(os.pathsep, ",").split(",")
        if path.strip()
    ]


def _pdfs_from_path(path):
    path = _normalize_path(path)
    if path.is_dir():
        return sorted(path.glob("*.pdf"))
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"O arquivo não é um PDF: {path}")
    return [path]


def resolve_document_paths(document_paths=None):
    if document_paths is None:
        document_paths = _configured_document_paths()

    if document_paths is None:
        pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))
    elif isinstance(document_paths, (str, Path)):
        pdf_files = _pdfs_from_path(document_paths)
    else:
        pdf_files = []
        for document_path in document_paths:
            pdf_files.extend(_pdfs_from_path(document_path))

    if not pdf_files:
        raise FileNotFoundError(f"Nenhum PDF foi encontrado em {DOCUMENTS_DIR}.")

    return sorted(dict.fromkeys(pdf_files))


def resolve_document_path(document_path=None):
    return resolve_document_paths(document_path)[0]


def _path_to_cache(path):
    path = Path(path)
    if not path.is_absolute():
        path = BASE_DIR / path

    try:
        return str(path.relative_to(BASE_DIR))
    except ValueError:
        return str(path)


def _read_store_cache():
    if not STORE_CACHE_PATH.exists():
        return {}

    try:
        data = json.loads(STORE_CACHE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

    document_paths = data.get("document_paths", [])
    if data.get("document_path"):
        document_paths.append(data["document_path"])

    return {
        "store_name": data.get("store_name"),
        "document_paths": sorted(set(document_paths)),
    }


def _read_cached_store_name():
    env_store_name = os.getenv("FILE_SEARCH_STORE_NAME")
    if env_store_name:
        return env_store_name

    return _read_store_cache().get("store_name")


def _save_store_cache(store_name, document_paths):
    data = {
        "store_name": store_name,
        "document_paths": sorted(
            {_path_to_cache(document_path) for document_path in document_paths}
        ),
    }
    STORE_CACHE_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _wait_for_indexing(client, operation):
    while not operation.done:
        print("Processando documento...")
        time.sleep(5)
        operation = client.operations.get(operation)

    if operation.error:
        raise RuntimeError(f"Falha ao indexar documento: {operation.error}")

    return operation


def upload_documents_to_store(client, store_name, document_paths):
    document_paths = resolve_document_paths(document_paths)
    uploaded_paths = []

    for index, document_path in enumerate(document_paths, start=1):
        print(f"Indexando {document_path.name} ({index}/{len(document_paths)})...")
        operation = client.file_search_stores.upload_to_file_search_store(
            file=document_path,
            file_search_store_name=store_name,
            config={
                "display_name": document_path.stem,
            },
        )
        _wait_for_indexing(client, operation)
        uploaded_paths.append(document_path)

    return uploaded_paths


def create_store_with_documents(client=None, document_paths=None):
    client = client or get_client()
    document_paths = resolve_document_paths(document_paths)
    store = client.file_search_stores.create(
        config={
            "display_name": STORE_DISPLAY_NAME,
            "embedding_model": EMBEDDING_MODEL,
        }
    )

    upload_documents_to_store(client, store.name, document_paths)
    _save_store_cache(store.name, document_paths)

    print("Documentos indexados.")
    print("Nome da base:", store.name)

    return store.name


def create_store_with_document(client=None, document_path=None):
    return create_store_with_documents(client=client, document_paths=document_path)


def get_or_create_store_name(client=None, document_paths=None, force_reindex=False):
    client = client or get_client()
    document_paths = resolve_document_paths(document_paths)

    if not force_reindex:
        env_store_name = os.getenv("FILE_SEARCH_STORE_NAME")
        if env_store_name:
            return env_store_name

        cache = _read_store_cache()
        cached_store_name = cache.get("store_name")
        if cached_store_name:
            cached_paths = set(cache.get("document_paths", []))
            current_paths = {_path_to_cache(document_path) for document_path in document_paths}
            missing_paths = [
                document_path
                for document_path in document_paths
                if _path_to_cache(document_path) not in cached_paths
            ]

            if missing_paths:
                upload_documents_to_store(client, cached_store_name, missing_paths)
                cached_paths.update(current_paths)

            _save_store_cache(cached_store_name, cached_paths)
            return cached_store_name

    return create_store_with_documents(client=client, document_paths=document_paths)


if __name__ == "__main__":
    get_or_create_store_name(force_reindex=False)
