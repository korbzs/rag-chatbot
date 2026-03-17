import logging
from langchain_ollama import OllamaEmbeddings
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from config import (
    CHROME_DB_PATH, EMBED_MODEL_NAME, TAVILY_TOP_K, CHROMA_TOP_K,
    GUARDRAIL_SYSTEM_PROMPT, ROUTER_SYSTEM_PROMPT,
    GRADER_SYSTEM_PROMPT, REWRITER_SYSTEM_PROMPT, GENERATOR_SYSTEM_PROMPT
)
from factory import ModelFactory
from src.security.filters import PromptInjectionFilter
from src.security.moderation import check_openai_moderation

logger = logging.getLogger(__name__)

class RouteDecision(BaseModel):
    datasource: str = Field(description="Route to 'vectorstore', 'web_search', 'both', or 'direct'.")

class GradeDecision(BaseModel):
    binary_score: str = Field(description="Relevance score 'yes' or 'no'")

class GuardrailDecision(BaseModel):
    decision: str = Field(description="Must be 'pass' or 'block'.")
    reason: str = Field(description="Reason for blocking, empty if pass.")

def guardrail_node(state: dict) -> dict:
    logger.info("GUARDRAIL AGENT")
    question = state.get("current_question", state.get("original_question", ""))
    
    security_filter = PromptInjectionFilter()
    if security_filter.detect_injection(question):
        return {"route": "blocked", "generation": "Security violation: Prompt injection detected."}
        
    if check_openai_moderation(question):
        return {"route": "blocked", "generation": "Security violation: Content moderation flagged."}
        
    sanitized_question = security_filter.sanitize_input(question)
    
###
# Ez 10mp is lehet LangSmith szerint, de fontosabb, hogy a user ne tudja pocsékolni az erőforrásunkat
    try:
        llm = ModelFactory.get_primary(streaming=False)
    except:
        ll = ModelFactory.get_fallback(streaming=False)

    llm_structured = llm.with_structured_output(GuardrailDecision)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", GUARDRAIL_SYSTEM_PROMPT),
        ("human", "{question}")
    ])
    
    result = (prompt | llm_structured).invoke({"question": sanitized_question})
    
    if result.decision.lower() == "block":
        return {"route": "blocked", "generation": f"Blocked: {result.reason}"}
        
###
    return {"route": "pass", "current_question": sanitized_question}

def router_node(state: dict) -> dict:
    logger.info("ROUTER AGENT")
    question = state.get("current_question", state.get("original_question", ""))
    
    try:
        llm = ModelFactory.get_primary(streaming=False)
    except:
        llm = ModelFactory.get_fallback(streaming=False)
    
    llm_structured = llm.with_structured_output(RouteDecision)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", ROUTER_SYSTEM_PROMPT),
        ("human", "{question}"),
    ])
    
    result = (prompt | llm_structured).invoke({"question": question})
    return {"route": result.datasource, "current_question": question}

def retriever_node(state: dict) -> dict:
    logger.info("RETRIEVER AGENT")
    question = state.get("current_question", "")
    
    if EMBED_MODEL_NAME or "nomic" in EMBED_MODEL_NAME:
        embeddings = OllamaEmbeddings(model=EMBED_MODEL_NAME)
    else:
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

    vectorstore = Chroma(persist_directory=CHROME_DB_PATH, embedding_function=embeddings)
    
    docs = vectorstore.similarity_search(
        question, 
        k=CHROMA_TOP_K
    )
    
    return {"documents": docs}

def web_search_node(state: dict) -> dict:
    logger.info("WEB SEARCH AGENT")
    search_query = state.get("current_question", "")
    tool = TavilySearchResults(max_results=TAVILY_TOP_K)
    
    try:
        raw_results = tool.invoke({"query": search_query})
    except Exception as e:
        logger.error(f"Tavily search failed: {e}")
        raw_results = []
    
    web_documents = []
    for result in raw_results:
        doc = Document(
            page_content=result.get("content", ""),
            metadata={"source": result.get("url", "Unknown Web Source")}
        )
        web_documents.append(doc)
    
    existing_docs = state.get("documents", [])
    
    combined_docs = existing_docs + web_documents
    
    return {"documents": combined_docs, "web_search_needed": False}

def relevance_grader_node(state: dict) -> dict:
    logger.info("RELEVANCE GRADER AGENT")
    question = state.get("original_question", state.get("current_question", ""))
    documents = state.get("documents", [])
    
    if not documents:
        return {"documents": [], "web_search_needed": True}
    
    # No inf cycle
    if len(documents) > 4:
        logger.info("Web search already performed, passing context to generation.")
        return {"documents": documents, "web_search_needed": False}
    
    try:
        llm = ModelFactory.get_primary(streaming=False)
    except:
        llm = ModelFactory.get_fallback(streaming=False)

    llm_structured = llm.with_structured_output(GradeDecision)
    
    # concat str -> 1 LLM call
    combined_context = "\n\n".join([d.page_content for d in documents])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", GRADER_SYSTEM_PROMPT),
        ("human", "Context: {context}\n\nQuestion: {question}"),
    ])
    
    score = (prompt | llm_structured).invoke({"question": question, "context": combined_context})
    
    needs_search = (score.binary_score.lower() == "no")
    if needs_search:
        logger.info("Context is incomplete. Web search required.")
    else:
        logger.info("Context is sufficient.")
        
    return {"documents": documents, "web_search_needed": needs_search}

def rewriter_node(state: dict) -> dict:
    logger.info("QUERY REWRITER AGENT")
    question = state.get("original_question", state.get("current_question", ""))
    
    llm = ModelFactory.get_primary(streaming=False)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", REWRITER_SYSTEM_PROMPT),
        ("human", "Question: {question}"),
    ])
    
    response = llm.invoke(prompt.format_messages(question=question))
    optimized_query = response.content.strip().replace('"', '').replace('\n', ' ')
    
    logger.info(f"Rewrote query to: {optimized_query}")
    retry_count = state.get("retry_count", 0) + 1
    return {"current_question": optimized_query, "retry_count": retry_count}

def generate_node(state: dict) -> dict:
    logger.info("GENERATION AGENT")
    
    question = state.get("original_question")
    if not question:
        question = state.get("current_question", "Question not found.")
        
    raw_documents = state.get("documents", [])
    
    unique_contents = set()
    context_parts = []
    
    for doc in raw_documents:
        content = getattr(doc, "page_content", str(doc))
        content_str = str(content)
        
        if content_str not in unique_contents:
            unique_contents.add(content_str)
            
            metadata = getattr(doc, "metadata", {})
            source = metadata.get("source", "Unknown Source")
            if "url" in metadata:
                source = metadata["url"]
                
            context_parts.append(f"--- SOURCE: {source} ---\n{content_str}")
    
    context = "\n\n".join(context_parts)
    
    llm = ModelFactory.get_primary(streaming=True)
    
    detected_lang = "hu" if any(c in question.lower() for c in "áéíóöőúüű") else "en" # ez kicsit hacky, system prompt elvileg lekezeli

    prompt = ChatPromptTemplate.from_messages([
        ("system", GENERATOR_SYSTEM_PROMPT),
        ("human",  """CRITICAL: You MUST answer in {language}! Do NOT use the language of the Context if it differs from {language}.

Original Question:
{question}

Context:
{context}"""),
    ])
    
    response = llm.invoke(prompt.format_messages(
        language=detected_lang,
        question=question,
        context=context
        ))

    return {"generation": response.content}