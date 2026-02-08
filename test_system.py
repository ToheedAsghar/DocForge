"""
Resume Test Script

This script will:
1. Load your resume from a folder
2. Ingest it into Pinecone
3. Let you ask questions about it

USAGE:
    python test_resume.py
"""

import os
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Main test function."""
    
    print("\n" + "="*60)
    print("📄 RESUME RAG SYSTEM TEST")
    print("="*60 + "\n")
    
    # ========== STEP 1: Check Environment ==========
    print("Step 1: Checking environment variables...")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found")
        print("   Please add it to your .env file")
        return
    
    if not os.getenv("PINECONE_API_KEY"):
        print("❌ Error: PINECONE_API_KEY not found")
        print("   Please add it to your .env file")
        return
    
    print("✅ Environment variables found\n")
    
    # ========== STEP 2: Get Resume Folder Path ==========
    resume_folder = input("Enter the path to your resume folder (or press Enter for './documents'): ").strip()
    
    if not resume_folder:
        resume_folder = "./documents"
    
    resume_path = Path(resume_folder)
    
    if not resume_path.exists():
        print(f"\n❌ Error: Folder '{resume_folder}' not found")
        print(f"   Please create it and add your resume file(s)")
        return
    
    print(f"\n✅ Found folder: {resume_folder}\n")
    
    # ========== STEP 3: Ingest Resume ==========
    print("Step 2: Ingesting your resume into Pinecone...")
    print("-" * 60)
    
    from backend.ingestion.pipeline import ingest_documents
    
    try:
        stats = ingest_documents(
            dir_path=resume_folder,
            chunk_size=800,  # Smaller chunks for resume (more precise)
            chunk_overlap=150,
            recursive=True
        )
        
        print("\n✅ Ingestion complete!")
        print(f"   Documents loaded: {stats['documents_loaded']}")
        print(f"   Chunks created: {stats['chunks_created']}")
        print(f"   Chunks uploaded to Pinecone: {stats['chunks_uploaded']}\n")
        
        if stats['chunks_uploaded'] == 0:
            print("⚠️  No chunks were uploaded. Please check if your resume file is in a supported format:")
            print("   Supported: .pdf, .docx, .txt, .md, .html")
            return
        
    except Exception as e:
        print(f"\n❌ Error during ingestion: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # ========== STEP 4: Test Questions ==========
    print("="*60)
    print("Step 3: Testing with sample questions...")
    print("="*60 + "\n")
    
    from backend.agents.graph import run_graph
    
    # Sample questions about resume
    test_questions = [
        # "Write a summary of the document.",
        # "What programming languages do I know?",
        # "What is my work experience?",
        # "What projects have I worked on?",
        # "What are my technical skills?"
    ]
    
    print("I'll ask a few questions about your resume to test the system.\n")
    
    for i, question in enumerate(test_questions, 1):
        print(f"\nQuestion {i}: {question}")
        print("-" * 60)
        
        try:
            result = run_graph(question)
            
            print(f"\nAnswer:")
            print(result['fact_checked_answer'])
            
            print(f"\n📊 Metadata:")
            print(f"   Query Type: {result['query_type']}")
            print(f"   Documents Retrieved: {len(result['retrieved_chunks'])}")
            print(f"   Validation: {'✅ PASSED' if result['validation_passed'] else '❌ FAILED'}")
            print(f"   Latency: {result['latency_ms']:.0f}ms")
            
            print("\n" + "="*60)
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            continue
    
    # ========== STEP 5: Interactive Mode ==========
    print("\n" + "="*60)
    print("Step 4: Interactive Q&A")
    print("="*60)
    print("\nNow you can ask your own questions!")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        try:
            question = input("\nYour question: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!\n")
                break
            
            print("\n🤔 Processing...\n")
            
            result = run_graph(question)
            
            print("Answer:")
            print(result['fact_checked_answer'])
            
            # Ask if they want details
            show_details = input("\nShow technical details? (y/n): ").strip().lower()
            
            if show_details == 'y':
                print(f"\n📊 Technical Details:")
                print(f"   Query Type: {result['query_type']}")
                print(f"   Confidence: {result.get('confidence', 'N/A')}")
                print(f"   Documents Retrieved: {len(result['retrieved_chunks'])}")
                print(f"   Validation: {'PASSED' if result['validation_passed'] else 'FAILED'}")
                # print(f"   Retry Count: {result['retry_count']}")
                print(f"   Total Latency: {result['latency_ms']:.0f}ms")
                print(f"   Tokens Used: {result.get('total_tokens_used', 'N/A')}")
                
                print(f"\n   Agent Workflow:")
                for step in result['agent_steps']:
                    print(f"      • {step['agent_name']}: {step['action']}")
                
                if result['retrieved_chunks']:
                    print(f"\n   Top Retrieved Chunks:")
                    for i, chunk in enumerate(result['retrieved_chunks'][:3], 1):
                        print(f"\n      [{i}] Score: {chunk['score']:.3f}")
                        print(f"          {chunk['text'][:150]}...")
            
            print("\n" + "-"*60)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # ========== Final Summary ==========
    print("\n" + "="*60)
    print("✅ TEST COMPLETE")
    print("="*60)
    print("\nYour resume RAG system is working!")
    print("\nWhat you can do now:")
    print("  1. Add more documents to your resume folder")
    print("  2. Re-run this script to update the index")
    print("  3. Ask any questions about your background")
    print("  4. Use this system in job applications or interviews")
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    # Check if .env file exists
    if not Path(".env").exists():
        print("\n⚠️  WARNING: .env file not found!")
        print("\nPlease create a .env file with:")
        print("OPENAI_API_KEY=your-key-here")
        print("PINECONE_API_KEY=your-key-here")
        print("PINECONE_ENVIRONMENT=us-east-1")
        print("PINECONE_INDEX_NAME=techdoc-intelligence\n")
        sys.exit(1)
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Run main test
    main()