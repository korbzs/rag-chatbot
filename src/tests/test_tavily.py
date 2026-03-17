import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

def test_tavily_search():
    if not TAVILY_API_KEY:
        print("[ERROR] TAVILY_API_KEY is missing from .env file.")
        return

    print("Testing Tavily Search API")
    
    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)
        query = "What are the latest advancements in Agentic RAG?"
        
        print(f"Executing search for: '{query}'\n")
        
        
        response = client.search(query=query, search_depth="basic")
        
        results = response.get("results", [])
        
        if not results:
            print("No results found.")
            return
            
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['title']}")
            print(f"   URL: {result['url']}")
            print(f"   Content Snippet: {result['content'][:150]}...\n")
            
        print("Tavily test successful.")
        
    except Exception as e:
        print(f"[ERROR] Tavily search failed: {e}")

if __name__ == "__main__":
    test_tavily_search()