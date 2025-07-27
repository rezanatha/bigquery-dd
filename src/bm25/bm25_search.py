from typing import List, Dict, Tuple
import pickle


def save_bm25_model(model, filepath: str):
    """Save BM25 model"""
    data = {
        "model": model,
    }
    with open(filepath, "wb") as f:
        pickle.dump(data, f)

    print(f"Saved BM25 model to {filepath}")


def print_bm25_results(results: List[Tuple[float, Dict]], query: str):
    """Pretty print BM25 search results"""
    print(f"\nBM25 Results for query: '{query}'")
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
        print(f"{i}. Score: {score:.4f}")
        print(f"   Table: {table_name}")

        columns = metadata.get("all_columns", "N/A")
        columns_display = f"{columns[:100]}{'...' if len(str(columns)) > 100 else ''}"
        print(f"   Columns: {columns_display}")
        print()


def load_bm25_model(filepath: str):
    """Load BM25"""
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    print(f"Loaded BM25 model from {filepath}")
    return model


def load_search_components():
    return load_bm25_model("models/bm25.pkl")["model"]


# if __name__ == '__main__':
# train()

# # Test searches
# test_queries = [
#     "Where I can find the daily campaign plan data?",
#     "help me find fraud related tables",
#     "security incidents",
# ]

# for query in test_queries:
#     results = bm25.search(query, top_k=5)
#     print_bm25_results(results, query)
#     print("-" * 40)
