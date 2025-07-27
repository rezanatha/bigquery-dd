from typing import List, Dict, Tuple, Set
from semantic import semantic_search
from semantic import EmbeddingGenerator  # noqa: F401 - Required for pickle loading


class HybridTableSearch:
    def __init__(self, semantic_search, bm25_search):
        """
        Initialize hybrid search with semantic + BM25

        Args:
            embedding_path: Path to embedding data (optional, for existing embeddings)
            bm25_data: DataFrame to build BM25 index from
        """
        # Load semantic search components
        self.embedding_data, self.generator = semantic_search

        # Initialize BM25
        self.bm25 = bm25_search

    def search_semantic(self, query: str, top_k: int = 50) -> List[Tuple[float, Dict]]:
        """Semantic search using embeddings"""
        if self.embedding_data is None:
            return []

        return semantic_search.search_similar_tables(
            query, self.embedding_data, self.generator, top_k=top_k
        )

    def search_bm25(self, query: str, top_k: int = 50) -> List[Tuple[float, Dict]]:
        """Keyword search using BM25"""
        return self.bm25.search(query, top_k=top_k)

    def reciprocal_rank_fusion(
        self,
        semantic_results: List[Tuple[float, Dict]],
        bm25_results: List[Tuple[float, Dict]],
        k: int = 60,
    ) -> List[Tuple[float, Dict]]:
        """
        Combine results using Reciprocal Rank Fusion (RRF)

        Args:
            semantic_results: Results from semantic search
            bm25_results: Results from BM25 search
            k: RRF parameter (typically 60, higher = more conservative)

        Returns:
            Combined results sorted by RRF score
        """

        def get_table_id(metadata: Dict) -> str:
            return (
                f"{metadata.get('table_catalog', 'N/A')}."
                f"{metadata.get('table_schema', 'N/A')}."
                f"{metadata.get('table_name', 'N/A')}"
            )

        # Create rank mappings
        semantic_ranks = {}
        for rank, (score, metadata) in enumerate(semantic_results, 1):
            table_id = get_table_id(metadata)
            semantic_ranks[table_id] = rank

        bm25_ranks = {}
        for rank, (score, metadata) in enumerate(bm25_results, 1):
            table_id = get_table_id(metadata)
            bm25_ranks[table_id] = rank

        # Get all unique tables
        all_tables: Set[str] = set(semantic_ranks.keys()) | set(bm25_ranks.keys())

        # Calculate RRF scores
        rrf_scores = {}
        table_metadata = {}

        # Store metadata from both sources
        for _, metadata in semantic_results:
            table_id = get_table_id(metadata)
            table_metadata[table_id] = metadata

        for _, metadata in bm25_results:
            table_id = get_table_id(metadata)
            if table_id not in table_metadata:  # Don't overwrite semantic metadata
                table_metadata[table_id] = metadata

        for table_id in all_tables:
            rrf_score = 0

            # Add semantic contribution
            if table_id in semantic_ranks:
                rrf_score += 1 / (k + semantic_ranks[table_id])

            # Add BM25 contribution
            if table_id in bm25_ranks:
                rrf_score += 1 / (k + bm25_ranks[table_id])

            rrf_scores[table_id] = rrf_score

        # Sort by RRF score and create results
        sorted_tables = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for table_id, rrf_score in sorted_tables:
            if table_id in table_metadata:
                results.append((rrf_score, table_metadata[table_id]))

        return results

    def search_hybrid(
        self, query: str, top_k: int = 10, rrf_k: int = 60, semantic_weight: float = 0.6
    ) -> List[Tuple[float, Dict]]:
        """
        Hybrid search using RRF + weighted scoring

        Args:
            query: Search query
            top_k: Number of final results
            rrf_k: RRF parameter
            semantic_weight: Weight for semantic vs keyword (0.6 = 60% semantic)

        Returns:
            Combined search results
        """
        # Get results from both methods (get more to ensure good fusion)
        semantic_results = self.search_semantic(query, top_k=50)
        bm25_results = self.search_bm25(query, top_k=50)

        # Use RRF to combine
        rrf_results = self.reciprocal_rank_fusion(
            semantic_results, bm25_results, k=rrf_k
        )

        # Optionally add weighted scoring on top of RRF
        if semantic_weight != 0.5:
            rrf_results = self._apply_weighted_boost(
                rrf_results, semantic_results, bm25_results, semantic_weight
            )

        return rrf_results[:top_k]

    def _apply_weighted_boost(
        self,
        rrf_results: List[Tuple[float, Dict]],
        semantic_results: List[Tuple[float, Dict]],
        bm25_results: List[Tuple[float, Dict]],
        semantic_weight: float,
    ) -> List[Tuple[float, Dict]]:
        """Apply additional weighting on top of RRF"""

        def get_table_id(metadata: Dict) -> str:
            return (
                f"{metadata.get('table_catalog', 'N/A')}."
                f"{metadata.get('table_schema', 'N/A')}."
                f"{metadata.get('table_name', 'N/A')}"
            )

        # Create score mappings
        semantic_scores = {}
        for score, metadata in semantic_results:
            table_id = get_table_id(metadata)
            semantic_scores[table_id] = score

        bm25_scores = {}
        for score, metadata in bm25_results:
            table_id = get_table_id(metadata)
            bm25_scores[table_id] = score

        # Normalize scores to 0-1
        if semantic_scores:
            max_sem = max(semantic_scores.values())
            min_sem = min(semantic_scores.values())
            if max_sem > min_sem:
                semantic_scores = {
                    k: (v - min_sem) / (max_sem - min_sem)
                    for k, v in semantic_scores.items()
                }

        if bm25_scores:
            max_bm25 = max(bm25_scores.values())
            min_bm25 = min(bm25_scores.values())
            if max_bm25 > min_bm25:
                bm25_scores = {
                    k: (v - min_bm25) / (max_bm25 - min_bm25)
                    for k, v in bm25_scores.items()
                }

        # Apply weighted boost to RRF scores
        boosted_results = []
        for rrf_score, metadata in rrf_results:
            table_id = get_table_id(metadata)

            # Get normalized scores
            sem_score = semantic_scores.get(table_id, 0)
            bm25_score = bm25_scores.get(table_id, 0)

            # Weighted combination
            weighted_score = (
                semantic_weight * sem_score + (1 - semantic_weight) * bm25_score
            )

            # Combine RRF with weighted score
            final_score = 0.7 * rrf_score + 0.3 * weighted_score

            boosted_results.append((final_score, metadata))

        # Re-sort by final score
        boosted_results.sort(key=lambda x: x[0], reverse=True)
        return boosted_results
