import os
from langchain_core.tools import tool
from tavily import TavilyClient

@tool
def web_search(query: str) -> str:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Error: TAVILY_API_KEY is not set."

    client = TavilyClient(api_key=api_key)
    
    try:
        response = client.search(query=query, search_depth="basic")
        results = response.get("results", [])
        
        if not results:
            return "No relevant web search results found."
            
        
        formatted_results = "\n\n".join(
            [f"Title: {r['title']}\nURL: {r['url']}\nContent: {r['content']}" for r in results]
        )
        return formatted_results
        
    except Exception as e:
        return f"Web search failed: {str(e)}"