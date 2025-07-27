import pickle
import os
from typing import List, Union, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingGenerator:
    def __init__(self, model=None):
        """Initialize with model loaded once"""
        self.model = model

    def generate_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for multiple texts efficiently"""
        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings

    def save_model(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            generator = pickle.load(f)
            self.model = generator.model

def load_embeddings(filepath: str):
    """Load saved embeddings"""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded {len(data['embeddings'])} embeddings from {filepath}")
    return data

def search_similar_tables(query: str, embedding_data: dict, generator: EmbeddingGenerator, top_k: int = 5) -> List[Tuple[float, dict]]:
    """
    Find most similar tables to a query string

    Args:
        query: Search query string
        embedding_data: Loaded embedding data with 'embeddings' and 'metadata'
        generator: EmbeddingGenerator instance
        top_k: Number of top results to return

    Returns:
        List of (similarity_score, table_metadata) tuples sorted by similarity
    """
    # Generate embedding for query
    query_embedding = generator.generate_batch([query])

    # Calculate cosine similarity with all table embeddings
    similarities = cosine_similarity(query_embedding, embedding_data['embeddings'])[0]

    # Get top k most similar
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        similarity_score = similarities[idx]
        table_metadata = embedding_data['metadata'][idx]
        results.append((similarity_score, table_metadata))

    return results

def print_search_results(results: List[Tuple[float, dict]], query: str):
    """Pretty print search results"""
    print(f"\nTop {len(results)} results for query: '{query}'")
    print("=" * 60)

    for i, (score, metadata) in enumerate(results, 1):
        print(f"{i}. Similarity: {score:.4f}")
        print(f"   Table: {metadata.get('table_catalog', 'N/A')}.{metadata.get('table_schema', 'N/A')}.{metadata.get('table_name', 'N/A')}")
        print(f"   Columns: {metadata.get('all_columns', 'N/A')[:100]}{'...' if len(str(metadata.get('all_columns', ''))) > 100 else ''}")
        print()

if __name__ == '__main__':
    # Load embeddings and generator
    embedding_data = load_embeddings('data/embedding/miniLM_embedding.pkl')
    generator = EmbeddingGenerator()
    generator.load_model('models/miniLM_embedding_generator.pkl')

    # Example search
    query = "customer orders transactions"
    results = search_similar_tables(query, embedding_data, generator, top_k=10)
    print_search_results(results, query)
