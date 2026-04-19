from langgraph.graph import StateGraph, END
from src.graph.state import GraphState
from src.nodes.agents import (
    guardrail_node,
    router_node,
    retriever_node,
    web_search_node,
    relevance_grader_node,
    rewriter_node,
    generate_node
)

def check_guardrail(state: GraphState):
    if state.get("route") == "blocked":
        return "blocked"
    return "pass"

def route_question(state: GraphState):
    route = state.get("route", "")
    if route == "web_search":
        return "web_search"
    elif route in ["vectorstore", "both"]:
        # Both routes start at the retriever, so router ---> web_search is not a must
        return "retriever"
    return "generate"

def after_retriever(state: GraphState):
    if state.get("route") == "both":
        return "web_search"
    return "relevance_grader"

def decide_to_generate(state: GraphState):
    if state.get("web_search_needed", False):
        if state.get("retry_count", 0) >= 3:
            return "generate"
        return "rewriter"
    return "generate"

workflow = StateGraph(GraphState)

workflow.add_node("guardrail", guardrail_node)
workflow.add_node("router", router_node)
workflow.add_node("retriever", retriever_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("relevance_grader", relevance_grader_node)
workflow.add_node("rewriter", rewriter_node)
workflow.add_node("generate", generate_node)

workflow.set_entry_point("guardrail")

workflow.add_conditional_edges("guardrail", check_guardrail, {
    "blocked": END,
    "pass": "router"
})

workflow.add_conditional_edges("router", route_question, {
    "web_search": "web_search",
    "retriever": "retriever",
    "generate": "generate"
})

workflow.add_conditional_edges("retriever", after_retriever, {
    "web_search": "web_search",
    "relevance_grader": "relevance_grader"
})

workflow.add_edge("web_search", "relevance_grader")

workflow.add_conditional_edges("relevance_grader", decide_to_generate, {
    "rewriter": "rewriter",
    "generate": "generate"
})

workflow.add_edge("rewriter", "web_search")
workflow.add_edge("generate", END)

app = workflow.compile()