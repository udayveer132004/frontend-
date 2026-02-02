
import logging
import sys
import time
from backend.chat.rag_engine import RAGEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_rag")

def test_rag_refactor():
    print(f"Python: {sys.executable}")
    # print(f"Path: {sys.path}")
    
    print("--- 0. Checking Imports ---")
    try:
        import sentence_transformers
        print(f"sentence_transformers: {sentence_transformers.__file__}")
        import langchain_huggingface
        print(f"langchain_huggingface: {langchain_huggingface.__file__}")
    except ImportError as e:
        print(f"Import Check Failed: {e}")
    
    print("--- 1. Initializing RAGEngine ---")
    try:
        rag = RAGEngine()
        print("SUCCESS: RAGEngine initialized")
    except Exception as e:
        print(f"FAIL: RAGEngine initialization failed: {e}")
        sys.exit(1)

    print("\n--- 2. Ingesting Text ---")
    sample_text = """
    Praveen is a software engineer with 3 years of experience.
    He knows Python, Qdrant, and LangChain.
    He built a resume parser project.
    """
    try:
        rag.ingest_text(sample_text, metadata={"source": "test"})
        print("SUCCESS: Text ingested")
    except Exception as e:
        print(f"FAIL: Ingestion failed: {e}")
        sys.exit(1)

    print("\n--- 3. Testing Query (Retrieval Only) ---")
    try:
        # Validating retrieval directly first
        docs = rag.vector_store.similarity_search("What does Praveen know?", k=2)
        if docs:
            print(f"SUCCESS: Retrieved {len(docs)} docs")
            print(f"Content: {docs[0].page_content}")
        else:
            print("FAIL: No docs retrieved")
            sys.exit(1)
    except Exception as e:
        print(f"FAIL: Retrieval failed: {e}")
        sys.exit(1)

    print("\n--- VERIFICATION COMPLETE: ALL CHECKS PASSED ---")

if __name__ == "__main__":
    test_rag_refactor()
