import pandas as pd
import json
import re
import unicodedata
from pathlib import Path
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/LaBSE")
QDRANT_PATH = "./qdrant_data"

qdrant_client = QdrantClient(path=QDRANT_PATH)
qdrant_client.recreate_collection(
    collection_name="transcript_search",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
)

vector_store = QdrantVectorStore(
    embedding=embeddings,
    client=qdrant_client,
    collection_name="transcript_search"
)

LANGUAGE_MAP = {
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "ml": "Malayalam"
}

def format_language(lang_code: str) -> str:
    return LANGUAGE_MAP.get(lang_code.lower(), lang_code)

def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_documents_from_path(path: str) -> list[Document]:
    path = Path(path)
    all_docs = []

    for file in path.glob("**/*"):
        if file.suffix == ".csv":
            df = pd.read_csv(file)
            for _, row in df.iterrows():
                if not all(k in row for k in ["transcript", "user_id", "language", "timestamp"]):
                    continue
                cleaned = clean_text(row["transcript"])
                lang_full = format_language(row["language"])
                all_docs.append(Document(
                    page_content=cleaned,
                    metadata={
                        "user_id": row["user_id"],
                        "language": row["language"],
                        "timestamp": row["timestamp"],
                        "source": str(file)
                    }
                ))
        elif file.suffix == ".json":
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                    if isinstance(data, list):
                        entries = data
                    elif isinstance(data, dict):
                        entries = [data]
                    else:
                        continue

                    for entry in entries:
                        if not all(k in entry for k in ["text", "user_id", "language", "timestamp"]):
                            continue
                        cleaned = clean_text(entry["text"])
                        lang_full = format_language(entry["language"])
                        all_docs.append(Document(
                            page_content=cleaned,
                            metadata={
                                "user_id": entry["user_id"],
                                "language": entry["language"],
                                "timestamp": entry["timestamp"],
                                "source": str(file)
                            }
                        ))
            except Exception as e:
                print(f"Failed to load {file}: {e}")

    return all_docs



# root_folder = "./transcripts"
documents=[]
documents += load_documents_from_path("./transcripts/hindi")
documents += load_documents_from_path("./transcripts/telgu")
documents += load_documents_from_path("./transcripts/tamil")
documents += load_documents_from_path("./transcripts/malayalam")

print(f"Adding {len(documents)} documents to Qdrant...")
ids = vector_store.add_documents(documents)
print("Added documents with IDs:", ids)