"""
Interactive Demo - Chat with Your Documents

A command-line chat interface for the TechDoc Intelligence system.
Ask questions, get answers, and see how the multi-agent system works.

FEATURES:
- Interactive Q&A with your documents
- Show detailed agent workflow
- Display system statistics
- Color-coded output (if terminal supports it)
- Command shortcuts

USAGE:
    python interactive_demo.py
"""

import os
import sys
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()


# ========== TERMINAL COLORS ==========

class Colors:
    """ANSI color codes for terminal output."""
    
    # Check if terminal supports colors
    ENABLED = sys.stdout.isatty()
    
    # Color codes
    RESET = '\033[0m' if ENABLED else ''
    BOLD = '\033[1m' if ENABLED else ''
    
    # Text colors
    RED = '\033[91m' if ENABLED else ''
    GREEN = '\033[92m' if ENABLED else ''
    YELLOW = '\033[93m' if ENABLED else ''
    BLUE = '\033[94m' if ENABLED else ''
    MAGENTA = '\033[95m' if ENABLED else ''
    CYAN = '\033[96m' if ENABLED else ''
    WHITE = '\033[97m' if ENABLED else ''
    GRAY = '\033[90m' if ENABLED else ''


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.RESET}\n")


def print_error(text: str):
    """Print an error message."""
    print(f"{Colors.RED}❌ {text}{Colors.RESET}")


def print_success(text: str):
    """Print a success message."""
    print(f"{Colors.GREEN}✅ {text}{Colors.RESET}")


def print_warning(text: str):
    """Print a warning message."""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.RESET}")


def print_info(text: str):
    """Print an info message."""
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.RESET}")


# ========== CHAT FUNCTIONS ==========

def check_setup() -> bool:
    """
    Verify system is set up and ready.
    
    Returns:
        True if ready, False otherwise
    """
    
    print_header("💬 TECHDOC INTELLIGENCE - Interactive Demo")
    
    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        print_error("OPENAI_API_KEY not found in environment")
        print("   Create a .env file with your API keys")
        return False
    
    if not os.getenv("PINECONE_API_KEY"):
        print_error("PINECONE_API_KEY not found in environment")
        print("   Create a .env file with your API keys")
        return False
    
    # Check vector store
    try:
        from backend.ingestion.pipeline import get_stats
        
        stats = get_stats()
        total_vectors = stats.get('total_vectors', 0)
        
        if total_vectors == 0:
            print_error("No documents found in vector store!")
            print("\n   To fix this:")
            print("   1. Run: python test_system.py")
            print("   2. Or: python -c \"from backend.ingestion import ingest_documents; ingest_documents('./your_docs/')\"")
            return False
        
        print_success(f"System ready with {total_vectors} document chunks")
        
    except Exception as e:
        print_error(f"Failed to connect to vector store: {str(e)}")
        return False
    
    return True


def show_help():
    """Display help information."""
    print(f"\n{Colors.BOLD}Available Commands:{Colors.RESET}")
    print(f"  {Colors.GREEN}help{Colors.RESET}       - Show this help message")
    print(f"  {Colors.GREEN}stats{Colors.RESET}      - Show system statistics")
    print(f"  {Colors.GREEN}clear{Colors.RESET}      - Clear the screen")
    print(f"  {Colors.GREEN}cache{Colors.RESET}      - Show cache statistics")
    print(f"  {Colors.GREEN}quit{Colors.RESET}       - Exit the demo (or: exit, q)")
    print(f"\n{Colors.BOLD}Tips:{Colors.RESET}")
    print(f"  - Ask any question about your documents")
    print(f"  - Type 'details' after an answer to see agent workflow")
    print(f"  - Answers are cached for faster repeated queries")
    print()


def show_stats():
    """Display system statistics."""
    try:
        from backend.ingestion.pipeline import get_stats
        from backend.services.cache import get_cache_service
        
        # Vector store stats
        vector_stats = get_stats()
        
        print(f"\n{Colors.BOLD}📊 System Statistics:{Colors.RESET}")
        print(f"{Colors.CYAN}Vector Store:{Colors.RESET}")
        print(f"   Total vectors: {vector_stats.get('total_vectors', 0)}")
        print(f"   Dimension: {vector_stats.get('dimension', 0)}")
        
        # Cache stats
        cache = get_cache_service()
        if cache.enabled:
            cache_stats = cache.get_stats()
            print(f"\n{Colors.CYAN}Cache:{Colors.RESET}")
            print(f"   Status: Enabled")
            print(f"   Cached queries: {cache_stats.get('cached_queries', 0)}")
            print(f"   Memory used: {cache_stats.get('memory_used_mb', 0):.2f} MB")
            print(f"   TTL: {cache_stats.get('ttl_seconds', 0)} seconds")
        else:
            print(f"\n{Colors.CYAN}Cache:{Colors.RESET}")
            print(f"   Status: Disabled")
        
        print()
        
    except Exception as e:
        print_error(f"Failed to get stats: {str(e)}")


def show_cache_stats():
    """Display cache-specific statistics."""
    try:
        from backend.services.cache import get_cache_service
        
        cache = get_cache_service()
        
        if not cache.enabled:
            print_warning("Cache is disabled")
            return
        
        stats = cache.get_stats()
        
        print(f"\n{Colors.BOLD}💾 Cache Statistics:{Colors.RESET}")
        print(f"   Cached queries: {stats.get('cached_queries', 0)}")
        print(f"   Memory used: {stats.get('memory_used_mb', 0):.2f} MB")
        print(f"   TTL: {stats.get('ttl_seconds', 0)} seconds")
        
        # Option to clear cache
        if stats.get('cached_queries', 0) > 0:
            response = input(f"\n{Colors.YELLOW}Clear cache? (y/n):{Colors.RESET} ").strip().lower()
            if response == 'y':
                count = cache.clear_all()
                print_success(f"Cleared {count} cached queries")
        
        print()
        
    except Exception as e:
        print_error(f"Failed to get cache stats: {str(e)}")


def process_query(query: str, show_details: bool = False) -> Optional[dict]:
    """
    Process a user query through the RAG system.
    
    Args:
        query: User's question
        show_details: Whether to show detailed workflow
        
    Returns:
        Result dictionary or None if failed
    """
    
    try:
        from backend.agents.graph import run_graph
        
        print(f"\n{Colors.GRAY}🤔 Thinking...{Colors.RESET}\n")
        
        # Run the query
        result = run_graph(query)
        
        # Display answer
        answer = result.get('fact_checked_answer', 'No answer generated')
        
        print(f"{Colors.BOLD}{Colors.BLUE}Assistant:{Colors.RESET}")
        print(f"{answer}\n")
        
        # Show metadata badge
        validation = "✅ Validated" if result.get('validation_passed') else "⚠️  Not validated"
        from_cache = "💾 Cached" if result.get('_from_cache') else "🔄 Fresh"
        
        print(f"{Colors.GRAY}{validation} | {from_cache} | {result.get('latency_ms', 0):.0f}ms | {len(result.get('retrieved_chunks', []))} docs{Colors.RESET}")
        
        # Auto-show details if validation failed
        if not result.get('validation_passed'):
            show_details = True
            print(f"\n{Colors.YELLOW}⚠️  Validation failed - showing workflow details{Colors.RESET}")
        
        # Show details if requested
        if show_details:
            show_query_details(result)
        
        return result
        
    except Exception as e:
        print_error(f"Query failed: {str(e)}")
        import traceback
        traceback.print_exc()
    return None

def show_query_details(result: dict):

    print(f"\n{Colors.BOLD} Query Details:{Colors.RESET}")

    # Basic info
    print(f"{Colors.CYAN}Query type:{Colors.RESET} {result.get('query_type', 'unknown')}")
    print(f"{Colors.CYAN}Documents retrieved:{Colors.RESET} {len(result.get('retrieved_chunks', []))}")
    print(f"{Colors.CYAN}Validation:{Colors.RESET} {'PASSED' if result.get('validation_passed') else 'FAILED'}")
    print(f"{Colors.CYAN}Retries:{Colors.RESET} {result.get('retry_count', 0)}")
    print(f"{Colors.CYAN}Time:{Colors.RESET} {result.get('latency_ms', 0):.0f}ms")
    print(f"{Colors.CYAN}Tokens:{Colors.RESET} {result.get('total_tokens_used', 0)}")

    # Validation issues
    if result.get('validation_issues'):
        print(f"\n{Colors.YELLOW}Validation issues:{Colors.RESET}")
        for issue in result['validation_issues'][:3]:
            print(f"   • {issue}")

    # Agent workflow
    if result.get('agent_steps'):
        print(f"\n{Colors.CYAN}Agent workflow:{Colors.RESET}")
        for i, step in enumerate(result['agent_steps'], 1):
            agent_name = step['agent_name'].replace('_', ' ').title()
            print(f"   {i}. {Colors.GREEN}{agent_name}{Colors.RESET}: {step['action']}")

    # Retrieved chunks
    if result.get('retrieved_chunks'):
        print(f"\n{Colors.CYAN}Top sources:{Colors.RESET}")
        for i, chunk in enumerate(result['retrieved_chunks'][:3], 1):
            filename = chunk.get('metadata', {}).get('filename', 'unknown')
            score = chunk.get('score', 0)
            print(f"   {i}. {filename} (score: {score:.3f})")

    print()

def chat_loop():
    """
    Main interactive chat loop.
    PROCESS:
    1. Get user input
    2. Check for commands
    3. Process query
    4. Display result
    5. Repeat
    """
    # Setup check
    if not check_setup():
        return

    # Show welcome message
    print(f"\n{Colors.GREEN}Ready to answer your questions!{Colors.RESET}")
    print(f"{Colors.GRAY}Type 'help' for commands, 'quit' to exit{Colors.RESET}\n")

    last_result = None

    while True:
        try:
            # Get input
            user_input = input(f"{Colors.BOLD}{Colors.MAGENTA}You:{Colors.RESET} ").strip()
            
            # Skip empty input
            if not user_input:
                continue
            
            # Check for commands
            command = user_input.lower()
            
            if command in ['quit', 'exit', 'q']:
                print(f"\n{Colors.GREEN}👋 Goodbye!{Colors.RESET}\n")
                break
            
            elif command == 'help':
                show_help()
                continue
            
            elif command == 'stats':
                show_stats()
                continue
            
            elif command == 'cache':
                show_cache_stats()
                continue
            
            elif command == 'clear':
                os.system('clear' if os.name == 'posix' else 'cls')
                print_header("💬 TECHDOC INTELLIGENCE - Interactive Demo")
                continue
            
            elif command == 'details' and last_result:
                show_query_details(last_result)
                continue
            
            # Process as query
            last_result = process_query(user_input)
            
        except KeyboardInterrupt:
            print(f"\n\n{Colors.GREEN}👋 Goodbye!{Colors.RESET}\n")
            break
        
        except EOFError:
            print(f"\n\n{Colors.GREEN}👋 Goodbye!{Colors.RESET}\n")
            break
        
        except Exception as e:
            print_error(f"Unexpected error: {str(e)}")
            import traceback
            traceback.print_exc()

def main():
    try:
        chat_loop()
    except Exception as e:
        print_error(f"Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()