from fastapi import FastAPI, Query
from search import get_rag_answer, QARequest
from query import get_similar_chunks

app = FastAPI()

@app.get("/search")
def search_endpoint(q: str = Query(..., description="Search query string")):
    return get_similar_chunks(q)

@app.get("/ask")
def ask_endpoint(q: str = Query(..., description="Question to ask the RAG system")):
    request = QARequest(query=q)
    return get_rag_answer(request)
