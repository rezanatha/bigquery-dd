import pandas as pd
from typing import List, Dict, Set
from semantic import semantic_search, EmbeddingGenerator  # noqa: F401
from bm25 import BM25TableSearch
from hybrid import HybridTableSearch
import numpy as np
import json


class SearchEvaluator:
    def __init__(self, test_file_path: str, search_type: str = "semantic"):
        """Initialize evaluator with test dataset

        Args:
            test_file_path: Path to test dataset CSV
            search_type: Type of search to evaluate ('semantic', 'bm25', 'hybrid')
        """
        self.test_data = pd.read_csv(test_file_path, sep="|")
        self.search_type = search_type

        # Initialize components based on search type
        if search_type in ["semantic", "hybrid"]:
            _, self.generator = semantic_search.load_search_components(
                use_embedding=False
            )
            self.test_embeddings, self.test_metadata = self._create_test_embeddings()
        else:
            self.generator = None
            self.test_embeddings = None
            self.test_metadata = self.test_data[
                ["table_catalog", "table_schema", "table_name", "all_columns"]
            ].to_dict("records")

        # Initialize BM25 for BM25 and hybrid search
        if search_type in ["bm25", "hybrid"]:
            self.bm25_searcher = BM25TableSearch()
            self.bm25_searcher.fit(self.test_data)
        else:
            self.bm25_searcher = None

        # Initialize hybrid searcher if needed
        if search_type == "hybrid":
            # Format embedding data like the semantic_search expects
            if self.test_embeddings is not None:
                embedding_data = {
                    "embeddings": self.test_embeddings,
                    "metadata": self.test_metadata,
                }
                semantic_components = (embedding_data, self.generator)
            else:
                semantic_components = (None, self.generator)
            self.hybrid_searcher = HybridTableSearch(
                semantic_components, self.bm25_searcher
            )
        else:
            self.hybrid_searcher = None

    def _create_test_embeddings(self):
        """Create embeddings for the test dataset"""
        print("Creating embeddings for test dataset...")

        def create_text(row):
            table_catalog = row["table_catalog"]
            schema = row["table_schema"]
            table = row["table_name"]
            columns = row["all_columns"]

            column_list = [col.strip() for col in columns.split(",")]

            text_parts = [
                f"Table catalog: {table_catalog}",
                f"Database schema: {schema}",
                f"Table name: {table}",
                f"Table: {schema}.{table}",
                f"Fields: {' '.join(column_list)}",
            ]

            return " | ".join(text_parts)

        self.test_data["text_form"] = self.test_data.apply(create_text, axis=1)
        texts = self.test_data["text_form"].tolist()
        metadata = self.test_data[
            ["table_catalog", "table_schema", "table_name", "all_columns"]
        ].to_dict("records")

        # Generate embeddings
        embeddings = self.generator.generate_batch(texts)

        print(f"Created embeddings for {len(texts)} test tables")
        return embeddings, metadata

    def calculate_relevance_score(
        self, table_keywords: Set[str], query_keywords: Set[str]
    ) -> float:
        """
        Calculate relevance score based on keyword matching
        - 0 if no keywords match
        - Base score of 1.0 for any match
        - +0.5 for each additional matching keyword
        """
        if not table_keywords or not query_keywords:
            return 0.0

        # Find intersection (case-insensitive)
        table_keywords_lower = {kw.strip().lower() for kw in table_keywords}
        query_keywords_lower = {kw.strip().lower() for kw in query_keywords}
        matches = table_keywords_lower.intersection(query_keywords_lower)

        if not matches:
            return 0.0

        # Base score of 1.0 + 0.5 for each additional match
        return 1.0 + (len(matches) - 1) * 0.5

    def get_ground_truth_relevance(self, query_keywords: Set[str]) -> Dict[str, float]:
        """Get relevance scores for all tables based on query keywords"""
        relevance_scores = {}

        for _, row in self.test_data.iterrows():
            table_id = (
                f"{row['table_catalog']}.{row['table_schema']}.{row['table_name']}"
            )
            table_keywords = set(row["relevant_keywords"].split(", "))
            relevance_score = self.calculate_relevance_score(
                table_keywords, query_keywords
            )
            relevance_scores[table_id] = relevance_score

        return relevance_scores

    def _search_test_data(self, query: str, top_k: int = 10):
        """Search within test data using the configured search method"""

        if self.search_type == "semantic":
            return self._search_test_embeddings(query, top_k)
        elif self.search_type == "bm25":
            return self.bm25_searcher.search(query, top_k)
        elif self.search_type == "hybrid":
            return self.hybrid_searcher.search_hybrid(query, top_k)
        else:
            raise ValueError(f"Unknown search type: {self.search_type}")

    def _search_test_embeddings(self, query: str, top_k: int = 10):
        """Search within test embeddings using cosine similarity"""
        from sklearn.metrics.pairwise import cosine_similarity

        # Check if embeddings are available
        if self.test_embeddings is None:
            raise ValueError(
                "Test embeddings not available. "
                "Make sure search_type includes semantic functionality."
            )

        if self.generator is None:
            raise ValueError("Embedding generator not available.")

        # Generate query embedding
        query_embedding = self.generator.generate_batch([query])

        # Ensure embeddings are numpy arrays with correct shapes
        if not isinstance(self.test_embeddings, np.ndarray):
            raise ValueError(
                f"test_embeddings must be numpy array, got {type(self.test_embeddings)}"
            )

        if not isinstance(query_embedding, np.ndarray):
            raise ValueError(
                f"query_embedding must be numpy array, got {type(query_embedding)}"
            )

        # Calculate cosine similarity
        similarities = cosine_similarity(query_embedding, self.test_embeddings)[0]

        # Get top k results
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Format results similar to search_similar_tables
        results = []
        for idx in top_indices:
            if idx < len(self.test_metadata):
                similarity_score = similarities[idx]
                metadata = self.test_metadata[idx]
                results.append((float(similarity_score), metadata))

        return results

    def evaluate_single_query(
        self, query: str, query_keywords: Set[str], top_k: int = 10
    ) -> Dict:
        """Evaluate a single search query"""
        # Get search results using configured search method
        search_results = self._search_test_data(query, top_k)

        # Get ground truth relevance scores
        ground_truth = self.get_ground_truth_relevance(query_keywords)

        # Calculate metrics
        relevant_found = 0
        total_relevance_score = 0
        reciprocal_rank = 0
        dcg = 0

        for rank, (similarity, metadata) in enumerate(search_results, 1):
            table_id = (
                f"{metadata.get('table_catalog')}."
                f"{metadata.get('table_schema')}."
                f"{metadata.get('table_name')}"
            )
            relevance = ground_truth.get(table_id, 0)

            if relevance > 0:
                relevant_found += 1
                total_relevance_score += relevance

                # Mean Reciprocal Rank - only count first relevant result
                if reciprocal_rank == 0:
                    reciprocal_rank = 1.0 / rank

            # DCG calculation
            if relevance > 0:
                dcg += relevance / np.log2(rank + 1)

        # Calculate ideal DCG (if we ranked perfectly)
        ideal_relevance = sorted(ground_truth.values(), reverse=True)[:top_k]
        idcg = sum(
            rel / np.log2(rank + 1)
            for rank, rel in enumerate(ideal_relevance, 1)
            if rel > 0
        )

        # Calculate metrics
        precision_at_k = relevant_found / top_k
        ndcg = dcg / idcg if idcg > 0 else 0
        avg_relevance = (
            total_relevance_score / relevant_found if relevant_found > 0 else 0
        )

        # Get total relevant tables in dataset
        total_relevant = sum(1 for score in ground_truth.values() if score > 0)
        recall_at_k = relevant_found / total_relevant if total_relevant > 0 else 0

        return {
            "query": query,
            "query_keywords": list(query_keywords),
            "total_results": len(search_results),
            "relevant_found": relevant_found,
            "total_relevant_in_dataset": total_relevant,
            "precision_at_k": precision_at_k,
            "recall_at_k": recall_at_k,
            "mrr": reciprocal_rank,
            "ndcg": ndcg,
            "avg_relevance_score": avg_relevance,
            "search_results": [
                {
                    "rank": rank,
                    "table_id": (
                        f"{metadata.get('table_catalog')}."
                        f"{metadata.get('table_schema')}."
                        f"{metadata.get('table_name')}"
                    ),
                    "similarity": float(similarity),
                    "relevance_score": ground_truth.get(
                        (
                            f"{metadata.get('table_catalog')}."
                            f"{metadata.get('table_schema')}."
                            f"{metadata.get('table_name')}"
                        ),
                        0,
                    ),
                }
                for rank, (similarity, metadata) in enumerate(search_results, 1)
            ],
        }

    def run_evaluation(self, test_queries: List[Dict[str, any]]) -> Dict:
        """Run evaluation on multiple queries"""
        results = []

        for query_info in test_queries:
            query = query_info["query"]
            keywords = set(query_info["keywords"])
            result = self.evaluate_single_query(query, keywords)
            results.append(result)

        # Calculate aggregate metrics
        avg_precision = np.mean([r["precision_at_k"] for r in results])
        avg_recall = np.mean([r["recall_at_k"] for r in results])
        avg_mrr = np.mean([r["mrr"] for r in results])
        avg_ndcg = np.mean([r["ndcg"] for r in results])
        avg_relevance = np.mean(
            [r["avg_relevance_score"] for r in results if r["avg_relevance_score"] > 0]
        )

        # F1 Score
        f1_score = (
            2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
            if (avg_precision + avg_recall) > 0
            else 0
        )

        return {
            "individual_results": results,
            "aggregate_metrics": {
                "average_precision_at_10": avg_precision,
                "average_recall_at_10": avg_recall,
                "average_mrr": avg_mrr,
                "average_ndcg": avg_ndcg,
                "average_relevance_score": avg_relevance,
                "f1_score": f1_score,
                "total_queries": len(results),
            },
        }

    def print_evaluation_report(self, evaluation_results: Dict):
        """Print a formatted evaluation report"""
        metrics = evaluation_results["aggregate_metrics"]

        print("SEARCH MODEL EVALUATION REPORT")
        print("=" * 50)
        print(f"Total Queries Tested: {metrics['total_queries']}")
        print()

        print("AGGREGATE METRICS:")
        print(f"  Precision@10:     {metrics['average_precision_at_10']:.3f}")
        print(f"  Recall@10:        {metrics['average_recall_at_10']:.3f}")
        print(f"  F1 Score:         {metrics['f1_score']:.3f}")
        print(f"  Mean Reciprocal Rank: {metrics['average_mrr']:.3f}")
        print(f"  NDCG:            {metrics['average_ndcg']:.3f}")
        print(f"  Avg Relevance:   {metrics['average_relevance_score']:.3f}")
        print()

        print("DETAILED RESULTS:")
        for result in evaluation_results["individual_results"]:
            print(f"\nQuery: '{result['query']}'")
            print(f"  Keywords: {result['query_keywords']}")
            print(
                f"  Found {result['relevant_found']}/"
                f"{result['total_relevant_in_dataset']} relevant tables"
            )
            print(
                f"  Precision: {result['precision_at_k']:.3f} | "
                f"Recall: {result['recall_at_k']:.3f}"
            )
            print(f"  MRR: {result['mrr']:.3f} | NDCG: {result['ndcg']:.3f}")

            # Show top 3 results
            print("  Top 3 results:")
            for res in result["search_results"][:3]:
                status = "H" if res["relevance_score"] > 0 else "L"
                print(
                    f"    {res['rank']}. {status} {res['table_id']} "
                    f"(sim: {res['similarity']:.3f}, rel: {res['relevance_score']:.1f})"
                )


if __name__ == "__main__":
    # Create test queries
    test_queries = [
        {
            "query": "Where I can find the daily campaign plan data?",
            "keywords": {"daily", "campaign", "plan"},
        },
        {"query": "detected fraud data", "keywords": {"fraud", "detection"}},
        {
            "query": "analytical data of our customer",
            "keywords": {"customer", "analytics"},
        },
        {"query": "find security-related data", "keywords": {"security"}},
        {
            "query": "we need all tables about compliance and audit",
            "keywords": {"compliance", "audit"},
        },
        {
            "query": "table about overall business performance",
            "keywords": {"performance", "business", "report"},
        },
        {"query": "data governance", "keywords": {"data", "governance"}},
        {
            "query": "where can I find employee monitoring data",
            "keywords": {"employee", "monitoring"},
        },
        {"query": "neywork security", "keywords": {"network", "security"}},
        {"query": "innovation our company has done", "keywords": {"innovation"}},
        {
            "query": "i want to know about our asset tracking",
            "keywords": {"asset", "tracking"},
        },
    ]

    # Evaluate all search methods
    search_methods = ["semantic", "bm25", "hybrid"]

    for method in search_methods:
        print(f"\n{'='*60}")
        print(f"EVALUATING {method.upper()} SEARCH")
        print(f"{'='*60}")

        # Initialize evaluator for this search method
        evaluator = SearchEvaluator(
            "data/raw/dataset_example_test.csv", search_type=method
        )

        # Run evaluation
        print(f"Running {method} evaluation...")
        results = evaluator.run_evaluation(test_queries)

        # Print report
        evaluator.print_evaluation_report(results)

        # Save results
        output_file = f"evals/evaluation_results_{method}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to {output_file}")

    print(f"\n{'='*60}")
    print("EVALUATION COMPARISON COMPLETE")
    print(f"{'='*60}")
    print("Check the evals/ directory for detailed results:")
    print("- evaluation_results_semantic.json")
    print("- evaluation_results_bm25.json")
    print("- evaluation_results_hybrid.json")
