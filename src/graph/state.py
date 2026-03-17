from typing import List, TypedDict
from langchain_core.documents import Document

class GraphState(TypedDict):
    original_question: str
    current_question: str
    generation: str
    documents: List[Document]
    route: str
    retry_count: int
    web_search_needed: bool