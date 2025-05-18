from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode

QDRANT_PATH = "./qdrant_data"
COLLECTION_NAME = "transcript_search"

qdrant_client = QdrantClient(path=QDRANT_PATH)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/LaBSE")

vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
    retrieval_mode=RetrievalMode.DENSE,
)
