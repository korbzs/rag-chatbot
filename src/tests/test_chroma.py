import os
import sys
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from config import CHROME_DB_PATH, EMBED_MODEL_NAME

def test_local_retrieval():
    
    absolute_db_path = os.path.join(PROJECT_ROOT, CHROME_DB_PATH.strip("./"))
    
    if not os.path.exists(absolute_db_path):
        print(f"Error: Database path {absolute_db_path} does not exist. Run ingestion first.")
        return

    print("Testing ChromaDB Retrieval")
    
    if "bge-m3" in EMBED_MODEL_NAME or "nomic" in EMBED_MODEL_NAME:
        embeddings = OllamaEmbeddings(model=EMBED_MODEL_NAME)
    else:
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    
    vectorstore = Chroma(persist_directory=absolute_db_path, embedding_function=embeddings)
    
    query = "What is the main topic of the ingested documents?"
    print(f"Executing search for: '{query}'\n")
    
    try:
        results = vectorstore.similarity_search_with_score(query, k=3)
        
        if not results:
            print("No results found. The database might be empty.")
            return
            
        for i, (doc, score) in enumerate(results, 1):
            print(f"Result {i} (Distance Score: {score:.4f})")
            print(f"Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"Language: {doc.metadata.get('language', 'Unknown')}")
            print(f"Content Snippet: {doc.page_content[:150]}...\n")
            
        print("ChromaDB test successful.")
        
    except Exception as e:
        print(f"ChromaDB search failed: {e}")

if __name__ == "__main__":
    test_local_retrieval()