from typing import List, Dict, Tuple
import pandas as pd
from semantic import semantic_search
from semantic import EmbeddingGenerator  # noqa: F401 - Required for pickle loading
from bm25 import bm25_search
from . import HybridTableSearch


def print_hybrid_results(results: List[Tuple[float, Dict]], query: str):
    """Pretty print hybrid search results"""
    print(f"\nHybrid Search Results for query: '{query}'")
    print("=" * 60)

    if not results:
        print("No results found.")
        return

    for i, (score, metadata) in enumerate(results, 1):
        table_name = (
            f"{metadata.get('table_catalog', 'N/A')}."
            f"{metadata.get('table_schema', 'N/A')}."
            f"{metadata.get('table_name', 'N/A')}"
        )
        print(f"{i}. RRF Score: {score:.4f}")
        print(f"   Table: {table_name}")

        columns = metadata.get("all_columns", "N/A")
        columns_display = f"{columns[:100]}{'...' if len(str(columns)) > 100 else ''}"
        print(f"   Columns: {columns_display}")
        print()


def compare_search_methods(
    query: str, hybrid_searcher: HybridTableSearch, top_k: int = 5
):
    """Compare different search methods side by side"""
    print(f"\n🔍 COMPARISON FOR QUERY: '{query}'")
    print("=" * 80)

    # Get results from each method
    semantic_results = hybrid_searcher.search_semantic(query, top_k)
    bm25_results = hybrid_searcher.search_bm25(query, top_k)
    hybrid_results = hybrid_searcher.search_hybrid(query, top_k)

    print(f"\n📊 SEMANTIC SEARCH (Top {len(semantic_results)}):")
    for i, (score, metadata) in enumerate(semantic_results, 1):
        table_name = (
            f"{metadata.get('table_schema', '')}.{metadata.get('table_name', '')}"
        )
        print(f"  {i}. {table_name} (score: {score:.3f})")

    print(f"\n🔤 BM25 SEARCH (Top {len(bm25_results)}):")
    for i, (score, metadata) in enumerate(bm25_results, 1):
        table_name = (
            f"{metadata.get('table_schema', '')}.{metadata.get('table_name', '')}"
        )
        print(f"  {i}. {table_name} (score: {score:.3f})")

    print(f"\n🎯 HYBRID RRF (Top {len(hybrid_results)}):")
    for i, (score, metadata) in enumerate(hybrid_results, 1):
        table_name = (
            f"{metadata.get('table_schema', '')}.{metadata.get('table_name', '')}"
        )
        print(f"  {i}. {table_name} (score: {score:.3f})")


# Example usage
if __name__ == "__main__":
    # Load test data
    data = pd.read_csv("data/raw/dataset_example.csv", sep="|")

    # Initialize hybrid search
    semantic_search = semantic_search.load_search_components()
    bm25_search = bm25_search.load_search_components()
    hybrid_searcher = HybridTableSearch(semantic_search, bm25_search)

    # Test queries
    test_queries = ["fraud detection", "customer analytics", "security incidents"]

    for query in test_queries:
        # Show comparison
        compare_search_methods(query, hybrid_searcher, top_k=5)

        # Show final hybrid results
        results = hybrid_searcher.search_hybrid(query, top_k=10)
        print_hybrid_results(results, query)
        print("-" * 80)
