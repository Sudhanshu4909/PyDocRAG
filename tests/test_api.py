#!/usr/bin/env python3
"""
Test script for PyDocRAG API
Simple, cross-platform way to test your RAG system
"""

import requests
import json
import sys
from datetime import datetime

API_URL = "http://localhost:8000"

def print_separator(char="=", length=60, color=None):
    """Print a separator line"""
    print(char * length)

def test_query(query: str, top_k: int = 5, library: str = None):
    """Query the RAG API and display results"""
    
    print_separator()
    print("PyDocRAG API Test")
    print_separator()
    print(f"Query: {query}")
    print(f"Top K: {top_k}")
    if library:
        print(f"Library: {library}")
    print()
    
    # Prepare request
    payload = {
        "query": query,
        "top_k": top_k
    }
    if library:
        payload["library"] = library
    
    # Make request
    try:
        print("Querying API...")
        response = requests.post(
            f"{API_URL}/api/v1/query",
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        
        # Display answer
        print()
        print_separator(color="green")
        print("ANSWER")
        print_separator()
        print(result['answer'])
        
        # Display sources
        print()
        print_separator()
        print(f"SOURCES ({len(result['sources'])} found)")
        print_separator()
        
        for i, source in enumerate(result['sources'], 1):
            print(f"\n[{i}] {source['library']} - {source['doc_type']}")
            print(f"    Title: {source['title'][:70]}...")
            print(f"    URL: {source['url']}")
            print(f"    Score: {source['score']:.3f}")
        
        # Display metadata
        print()
        print_separator()
        print("METADATA")
        print_separator()
        print(f"Response Time: {result['response_time']:.2f} seconds")
        print(f"Cached: {result.get('cached', False)}")
        
        # Save option
        print()
        save = input("Save full response to JSON file? (y/n): ")
        if save.lower() == 'y':
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rag_response_{timestamp}.json"
            with open(filename, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"✓ Saved to: {filename}")
        
        print()
        print_separator()
        
        return result
        
    except requests.exceptions.ConnectionError:
        print("\n✗ Error: Could not connect to API")
        print("Make sure the API is running:")
        print("  uvicorn api.main:app --reload")
        sys.exit(1)
    except requests.exceptions.Timeout:
        print("\n✗ Error: Request timed out")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"\n✗ HTTP Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)

def test_health():
    """Test the health endpoint"""
    try:
        response = requests.get(f"{API_URL}/health")
        response.raise_for_status()
        data = response.json()
        
        print("Health Check:")
        print(f"  Status: {data['status']}")
        print(f"  Version: {data['version']}")
        if 'collection_info' in data:
            info = data['collection_info']
            print(f"  Documents: {info.get('points_count', 'N/A')}")
        return True
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def interactive_mode():
    """Interactive query mode"""
    print("Interactive Mode - Type 'quit' to exit")
    print_separator()
    
    while True:
        print()
        query = input("Enter your query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not query:
            print("Please enter a query")
            continue
        
        test_query(query)

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test PyDocRAG API')
    parser.add_argument('query', nargs='?', help='Query string')
    parser.add_argument('-k', '--top-k', type=int, default=5, help='Number of results')
    parser.add_argument('-l', '--library', help='Filter by library')
    parser.add_argument('-i', '--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--health', action='store_true', help='Health check only')
    
    args = parser.parse_args()
    
    if args.health:
        test_health()
    elif args.interactive:
        interactive_mode()
    elif args.query:
        test_query(args.query, args.top_k, args.library)
    else:
        # Default query
        test_query("How to use pandas DataFrame?")

if __name__ == "__main__":
    main()