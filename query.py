from qdrant import vector_store
from search import extract_filters, build_qdrant_filter

def get_similar_chunks(query: str, k: int = 5):
    filters = extract_filters(query)
    qdrant_filter = build_qdrant_filter(filters)

    found_docs = vector_store.similarity_search(
        query=query,
        k=k,
        filter=qdrant_filter
    )

    return [
        {
            "text": doc.page_content,
            "metadata": doc.metadata
        }
        for doc in found_docs
    ]