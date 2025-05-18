from pydantic import BaseModel
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import re

from qdrant import vector_store
from qdrant_client import models

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

main_llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    openai_api_key=OPENAI_API_KEY,
    temperature=0.7,
    top_p=1.0,
    streaming=True    
)

rewrite_llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    openai_api_key=OPENAI_API_KEY,
    temperature=0.3
)

def rewrite_query_with_gpt(original_query: str) -> str:
    system_prompt = (
        """
You are a helpful assistant that rewrites user queries into a clear, formal, and standardized format for search and retrieval.

Follow these guidelines:
- Simplify informal or idiomatic phrases into formal, searchable language.
- Preserve the original intent of the query.
- If the query refers to a user by numeric ID (e.g., "user id 203"), rewrite it in the format: "User ID: user_203".
- If a language is mentioned (e.g., Hindi, Telugu), ensure it's spelled in full (not as codes like 'hi' or 'te').
- If a date or time reference is mentioned (e.g., "this week", "May 2024", "on 10th March"), convert it to ISO 8601 format if possible, or leave as a normalized date phrase for filtering (e.g., "between 2024-05-01 and 2024-05-07").

Only rewrite the query. Do not add explanations or extra context.
"""
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": original_query}
    ]

    response = rewrite_llm.invoke(messages)
    return response.content.strip()


LANGUAGE_MAP_REV = {
    "hi": "hindi",
    "ta": "tamil",
    "te": "telgu",
    "ml": "malayalam",
    "N/A": "N/A"
}
LANGUAGE_MAP = {
    "hindi": "hi",
    "tamil": "ta",
    "telgu": "te",
    "malayalam": "ml",
    "N/A": "N/A"
}

def extract_filters(query: str) -> dict:
    filters = {}

    user_match = re.search(r"user\s*id[:\s]*([0-9]+)", query, re.IGNORECASE)
    if user_match:
        filters["metadata.user_id"] = f"user_{user_match.group(1)}"

    lang_match = re.search(r"\b(Hindi|Tamil|Telugu|Malayalam)\b", query, re.IGNORECASE)
    if lang_match:
        lang_code = LANGUAGE_MAP.get(lang_match.group(1).lower())
        filters["metadata.language"] = lang_code

    print("filters: ", filters)

    return filters


prompt_template = """
You are an assistant helping with analyzing user conversations.
Use ONLY the transcript chunks provided below as your reference.
If the answer is clearly implied or can be reasonably inferred from the context, answer accordingly.
If the query asks about which user answer with the "User ID" provided in context.
If the context does not provide enough information, respond with "I don't know."

Context:
{context}

Question:
{question}

Answer:
"""

qa_prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)


class QARequest(BaseModel):
    query: str

def build_qdrant_filter(filters: dict) -> models.Filter:
    conditions = []
    for key, value in filters.items():
        if value is not None:
            conditions.append(
                models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=value)
                )
            )
    if not conditions:
        return None
    return models.Filter(must=conditions)


def get_rag_answer(request: QARequest):
    # rewritten_query = rewrite_query_with_gpt(request.query)
    rewritten_query = request.query

    filters = extract_filters(rewritten_query)
    qdrant_filter = build_qdrant_filter(filters)
    print("Qdrant filter:", qdrant_filter)

    search_kwargs = {"k": 10}
    if qdrant_filter:
        search_kwargs["filter"] = qdrant_filter

    retriever = vector_store.as_retriever(search_kwargs=search_kwargs)

    docs = retriever.get_relevant_documents(rewritten_query)

    for doc in docs:
        doc.page_content = (
            f"User ID: {doc.metadata.get('user_id', 'N/A')}\n"
            f", Timestamp: {doc.metadata.get('timestamp', 'N/A')}\n"
            f", Language: {LANGUAGE_MAP_REV.get(doc.metadata.get('language', 'N/A'))}\n"
            f", Transcript: {doc.page_content}"
        )

    context = "\n\n".join([doc.page_content for doc in docs])
    final_prompt = qa_prompt.format(context=context, question=rewritten_query)

    answer = main_llm.invoke(final_prompt)

    return {
        "original_query": request.query,
        "rewritten_query": rewritten_query,
        "filters_used": filters,
        "answer": answer.content,
        "sources": [
            {"text": doc.page_content, "metadata": doc.metadata}
            for doc in docs
        ]
    }
