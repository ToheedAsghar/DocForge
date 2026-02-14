"""
document Test Script

This script will:
1. Load your document from a folder
2. Ingest it into Pinecone
3. Let you ask questions about it

USAGE:
    python test_document.py
"""

import os
import sys
import asyncio
from pathlib import Path
from backend.logger import get_logger
from backend.spinner import Spinner

logger = get_logger(__name__)

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))


def main():
    """Main test function."""
    
    print("\n" + "="*60)
    print("DOCFORGE")
    print("="*60 + "\n")
    
    # # ========== STEP 1: Check Environment ==========
    # print("\nStep 1: Checking environment variables...")
    
    llm_provider = os.getenv("LLM_PROVIDER", "gpt").lower()
    if llm_provider == "gpt" and not os.getenv("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY not found")
        print("   Please add it to your .env file")
        return
    elif llm_provider == "gemini" and not os.getenv("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY not found")
        print("   Please add it to your .env file")
        return
    
    if not os.getenv("PINECONE_API_KEY"):
        print("Error: PINECONE_API_KEY not found")
        print("   Please add it to your .env file")
        return
    
    logger.info("Environment variables found")
    
    # ========== STEP 2: Get document Folder Path ==========
    documents_folder = input("Enter the path to your document folder (or press Enter for './documents'): ").strip()
    
    if not documents_folder:
        documents_folder = "./documents"
    
    documents_path = Path(documents_folder)
    
    if not documents_path.exists():
        print(f"\nFolder '{documents_folder}' not found, Please create it and add your document file(s)")
        return
    
    logger.info(f"Found folder: {documents_folder}")
    
    # ========== STEP 3: Ingest document ==========
    # print("\nStep 2: Ingesting your document into Pinecone...")
    
    from backend.ingestion.pipeline import ingest_documents
    
    try:
        with Spinner("Ingesting documents into Pinecone...", style="dots"):
            stats = ingest_documents(
                dir_path=documents_folder,
                chunk_size=800,  # Smaller chunks for document (more precise)
                chunk_overlap=150,
                recursive=True
            )
        
        # print("\nIngestion complete!")
        logger.info(f"Documents loaded: {stats['documents_loaded']}")
        logger.info(f"Chunks created: {stats['chunks_created']}")
        logger.info(f"Chunks uploaded to Pinecone: {stats['chunks_uploaded']}")
        
        if stats['chunks_uploaded'] == 0:
            print("No chunks were uploaded. Please check if your document file is in a supported format:")
            print("   Supported: .pdf, .docx, .txt, .md, .html")
            return
        
    except Exception as e:
        print(f"\nError during ingestion: {str(e)}")
        logger.error(f"Ingestion error: {str(e)}")
        return
    
    # ========== STEP 4: Test Questions ==========
    # print("\nStep 3: Testing with sample questions...\n")
    
    from backend.agents.graph import run_graph
    
    # # Sample questions about document
    # test_questions = [
    #     # "Write a summary of the document.",
    #     # "What programming languages do I know?",
    #     # "What is my work experience?",
    #     # "What projects have I worked on?",
    #     # "What are my technical skills?"
    # ]
    
    # print("I'll ask a few questions about your document to test the system.\n")
    
    # for i, question in enumerate(test_questions, 1):
    #     print(f"\nQuestion {i}: {question}")
        
    #     try:
    #         result = asyncio.run(run_graph(question))
            
    #         print(f"\nAnswer:")
    #         print(result['fact_checked_answer'])
            
    #         logger.info(f"Query Type: {result['query_type']}")
    #         logger.info(f"Documents Retrieved: {len(result['retrieved_chunks'])}")
    #         logger.info(f"Validation: {'PASSED' if result['validation_passed'] else 'FAILED'}")
    #         logger.info(f"Latency: {result['latency_ms']:.0f}ms")
            
    #     except Exception as e:
    #         print(f"Error: {str(e)}")
    #         continue
    
    # ========== STEP 5: Interactive Mode ==========
    # print("\nStep 4: Interactive Q&A")
    # print("Now you can ask your own questions!")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        try:
            question = input("\nUSER\t> ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!\n")
                break
            
            with Spinner("Thinking...", style="default"):
                result = asyncio.run(run_graph(question))
            
            print()  # blank line after spinner clears
            
            print(f"FORGE\t> {result['fact_checked_answer']}")
            
            # Ask if they want details
            show_details = input("\nShow technical details? (y/n): ").strip().lower()
            
            if show_details == 'y':
                print(f"\nTechnical Details:")
                print(f"   Query Type: {result['query_type']}")
                print(f"   Confidence: {result.get('confidence', 'N/A')}")
                print(f"   Documents Retrieved: {len(result['retrieved_chunks'])}")
                print(f"   Validation: {'PASSED' if result['validation_passed'] else 'FAILED'}")
                print(f"   Total Latency: {result['latency_ms']:.0f}ms")
                print(f"   Tokens Used: {result.get('total_tokens_used', 'N/A')}")
                
                print(f"\n   Agent Workflow:")
                for step in result['agent_steps']:
                    print(f"      {step['agent_name']}: {step['action']}")
                
                if result['retrieved_chunks']:
                    print(f"\n   Top Retrieved Chunks:")
                    for i, chunk in enumerate(result['retrieved_chunks'][:3], 1):
                        print(f"\n      [{i}] Score: {chunk['score']:.3f}")
                        print(f"          {chunk['text'][:150]}...")
            
            print("\n" + "-"*60)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!\n")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            logger.error(f"Interactive query error: {str(e)}")
            continue
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    print("\nYour RAG system is working!")
    print("\nWhat you can do now:")
    print("  1. Add more documents to your documents folder")
    print("  2. Re-run this script to update the index")
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    # Check if .env file exists
    if not Path(".env").exists():
        print("\nWARNING: .env file not found!")
        print("\nPlease create a .env file with:")
        print("OPENROUTER_API_KEY=your-key-here")
        print("PINECONE_API_KEY=your-key-here")
        print("PINECONE_ENVIRONMENT=us-east-1")
        print("PINECONE_INDEX_NAME=techdoc-intelligence\n")
        sys.exit(1)
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Run main test
    main()