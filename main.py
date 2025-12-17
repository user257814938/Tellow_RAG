import sys
from rag_system import RAGSystem

def main():
    rag = RAGSystem()
    
    # Default source for demonstration if none provided
    default_url = "https://python.langchain.com/docs/integrations/document_loaders/docling/"
    
    print("=== Simple RAG System ===")
    source = input(f"Enter document URL or path (default: {default_url}): ").strip()
    if not source:
        source = default_url

    try:
        documents = rag.load_documents(source)
        rag.setup_vector_store(documents)
        
        print("\nSystem ready! Ask a question (type 'exit' to quit).")
        while True:
            query = input("\nQuestion: ").strip()
            if query.lower() in ['exit', 'quit']:
                break
            if not query:
                continue
                
            print("Thinking...")
            answer = rag.query(query)
            print(f"\nAnswer: {answer}")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
