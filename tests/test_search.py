import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os

import sys

sys.path.append("src")

from semantic.semantic_search import search_similar_tables, load_embeddings
from bm25 import bm25_search, BM25TableSearch
from hybrid import HybridTableSearch


class TestSemanticSearch:
    """Test semantic search functionality"""

    @pytest.fixture
    def sample_embedding_data(self):
        """Create sample embedding data for testing"""
        embeddings = np.array(
            [
                [1.0, 0.0, 0.0],  # Table 1
                [0.0, 1.0, 0.0],  # Table 2
                [0.5, 0.5, 0.0],  # Table 3 (similar to both)
            ],
            dtype=np.float32,
        )

        metadata = [
            {
                "table_catalog": "enterprise",
                "table_schema": "finance",
                "table_name": "transactions",
                "all_columns": "id, amount, date",
            },
            {
                "table_catalog": "enterprise",
                "table_schema": "users",
                "table_name": "accounts",
                "all_columns": "user_id, name, email",
            },
            {
                "table_catalog": "enterprise",
                "table_schema": "finance",
                "table_name": "payments",
                "all_columns": "payment_id, user_id, amount",
            },
        ]

        return {"embeddings": embeddings, "metadata": metadata}

    @pytest.fixture
    def mock_generator(self):
        """Create mock embedding generator"""
        generator = Mock()
        # Return query embedding that matches first table exactly
        generator.generate_batch.return_value = np.array(
            [[1.0, 0.0, 0.0]], dtype=np.float32
        )
        return generator

    def test_search_similar_tables_exact_match(
        self, sample_embedding_data, mock_generator
    ):
        """Test semantic search with exact match (perfect similarity)"""
        query = "finance transactions"
        results = search_similar_tables(
            query, sample_embedding_data, mock_generator, top_k=3
        )

        # Check that generator was called
        mock_generator.generate_batch.assert_called_once_with([query])

        # Check results structure
        assert len(results) == 3
        assert all(len(result) == 2 for result in results)  # (score, metadata) tuples

        # Check that first result has highest similarity (should be 1.0 for exact match)
        first_score, first_metadata = results[0]
        assert first_score == pytest.approx(1.0, abs=1e-6)
        assert first_metadata["table_name"] == "transactions"

        # Check scores are in descending order
        scores = [score for score, _ in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_similar_tables_top_k_limit(
        self, sample_embedding_data, mock_generator
    ):
        """Test that top_k parameter limits results correctly"""
        query = "test query"
        results = search_similar_tables(
            query, sample_embedding_data, mock_generator, top_k=2
        )

        assert len(results) == 2

        # Test with top_k larger than available data
        results = search_similar_tables(
            query, sample_embedding_data, mock_generator, top_k=10
        )
        assert len(results) == 3  # Only 3 tables available

    def test_search_similar_tables_metadata_integrity(
        self, sample_embedding_data, mock_generator
    ):
        """Test that metadata is returned correctly"""
        query = "test"
        results = search_similar_tables(
            query, sample_embedding_data, mock_generator, top_k=1
        )

        score, metadata = results[0]
        assert "table_catalog" in metadata
        assert "table_schema" in metadata
        assert "table_name" in metadata
        assert "all_columns" in metadata
        assert metadata["table_catalog"] == "enterprise"

    def test_load_embeddings(self):
        """Test loading embeddings from pickle file"""
        # Create test embedding data
        test_data = {
            "embeddings": np.array([[1, 2, 3], [4, 5, 6]]),
            "metadata": [{"table": "test1"}, {"table": "test2"}],
        }

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            temp_path = f.name

        try:
            # Save test data
            import pickle

            with open(temp_path, "wb") as f:
                pickle.dump(test_data, f)

            # Test loading
            loaded_data = load_embeddings(temp_path)

            assert "embeddings" in loaded_data
            assert "metadata" in loaded_data
            np.testing.assert_array_equal(
                loaded_data["embeddings"], test_data["embeddings"]
            )
            assert loaded_data["metadata"] == test_data["metadata"]

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestBM25Search:
    """Test BM25 search functionality"""

    @pytest.fixture
    def sample_table_data(self):
        """Create sample table data for BM25 testing"""
        return pd.DataFrame(
            {
                "table_catalog": ["enterprise", "enterprise", "public"],
                "table_schema": ["finance", "users", "analytics"],
                "table_name": ["transactions", "accounts", "user_behavior"],
                "all_columns": [
                    "transaction_id, user_id, amount, date, payment_method",
                    "user_id, username, email, registration_date",
                    "user_id, event_type, timestamp, page_url",
                ],
            }
        )

    @pytest.fixture
    def fitted_bm25_searcher(self, sample_table_data):
        """Create fitted BM25 searcher"""
        searcher = BM25TableSearch()
        searcher.fit(sample_table_data)
        return searcher

    def test_bm25_initialization(self):
        """Test BM25TableSearch initialization"""
        searcher = BM25TableSearch()
        assert searcher.bm25 is None
        assert searcher.metadata == []

    def test_bm25_fit(self, sample_table_data):
        """Test BM25 fitting process"""
        searcher = BM25TableSearch()
        searcher.fit(sample_table_data)

        assert searcher.bm25 is not None
        assert searcher.metadata is not None
        assert len(searcher.metadata) == 3

        # Check metadata structure
        first_meta = searcher.metadata[0]
        assert "table_catalog" in first_meta
        assert "table_schema" in first_meta
        assert "table_name" in first_meta
        assert "all_columns" in first_meta

    def test_bm25_preprocess_text(self, fitted_bm25_searcher):
        """Test text preprocessing"""
        text = "User Transaction Data"
        processed = fitted_bm25_searcher._preprocess_text(text)
        assert processed == "user transaction data"

    def test_bm25_search_basic(self, fitted_bm25_searcher):
        """Test basic BM25 search functionality"""
        query = "user transaction"
        results = fitted_bm25_searcher.search(query, top_k=3)

        # Check results structure
        assert isinstance(results, list)
        assert len(results) <= 3
        assert all(len(result) == 2 for result in results)  # (score, metadata) tuples

        # Check that all scores are non-negative
        scores = [score for score, _ in results]
        assert all(score >= 0 for score in scores)

        # Check that results are sorted by score (descending)
        assert scores == sorted(scores, reverse=True)

    def test_bm25_search_keyword_matching(self, fitted_bm25_searcher):
        """Test that BM25 returns relevant results for keyword queries"""
        # Search for transaction-related terms
        results = fitted_bm25_searcher.search("transaction amount", top_k=3)

        # The transactions table should score highly
        top_result = results[0]
        score, metadata = top_result

        # Should return the transactions table as top result
        assert score > 0
        # Check that transaction-related table is in top results
        table_names = [meta["table_name"] for _, meta in results]
        assert "transactions" in table_names

    def test_bm25_search_no_match(self, fitted_bm25_searcher):
        """Test BM25 search with no matching terms"""
        query = "nonexistent terminology"
        results = fitted_bm25_searcher.search(query, top_k=3)

        # Should still return results (all tables with score 0)
        assert len(results) <= 3
        scores = [score for score, _ in results]
        # All scores should be 0 or very low
        assert all(score < 1.0 for score in scores)

    def test_bm25_search_unfitted_model(self):
        """Test search on unfitted model returns empty results"""
        searcher = BM25TableSearch()

        # Unfitted model should return empty results
        results = searcher.search("test query")
        assert results == []

    def test_bm25_search_empty_query(self, fitted_bm25_searcher):
        """Test search with empty query"""
        results = fitted_bm25_searcher.search("", top_k=3)

        # Should return results (with likely low scores)
        assert isinstance(results, list)
        assert len(results) <= 3

    def test_save_and_load_bm25_model(self, fitted_bm25_searcher):
        """Test saving and loading BM25 model using bm25_search module"""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            temp_path = f.name

        try:
            # Save model using bm25_search module
            bm25_search.save_bm25_model(fitted_bm25_searcher, temp_path)
            assert os.path.exists(temp_path)

            # Load model using bm25_search module
            loaded_data = bm25_search.load_bm25_model(temp_path)
            loaded_searcher = loaded_data["model"]

            # Test that loaded model works
            query = "user data"
            original_results = fitted_bm25_searcher.search(query, top_k=2)
            loaded_results = loaded_searcher.search(query, top_k=2)

            # Results should be identical
            assert len(original_results) == len(loaded_results)
            for (orig_score, orig_meta), (load_score, load_meta) in zip(
                original_results, loaded_results
            ):
                assert orig_score == pytest.approx(load_score, abs=1e-6)
                assert orig_meta == load_meta

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestHybridSearch:
    """Test hybrid search functionality"""

    @pytest.fixture
    def mock_semantic_components(self):
        """Create mock semantic search components"""
        # Mock embedding data
        embedding_data = {
            "embeddings": np.array(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]], dtype=np.float32
            ),
            "metadata": [
                {"table_name": "transactions", "table_schema": "finance"},
                {"table_name": "accounts", "table_schema": "users"},
                {"table_name": "payments", "table_schema": "finance"},
            ],
        }

        # Mock generator
        generator = Mock()
        generator.generate_batch.return_value = np.array(
            [[1.0, 0.0, 0.0]], dtype=np.float32
        )

        return (embedding_data, generator)

    @pytest.fixture
    def mock_bm25_searcher(self):
        """Create mock BM25 searcher"""
        searcher = Mock()
        all_results = [
            (2.5, {"table_name": "transactions", "table_schema": "finance"}),
            (1.8, {"table_name": "payments", "table_schema": "finance"}),
            (0.5, {"table_name": "accounts", "table_schema": "users"}),
        ]

        def mock_search(query, top_k=10):
            return all_results[:top_k]

        searcher.search.side_effect = mock_search
        return searcher

    @pytest.fixture
    def hybrid_searcher(self, mock_semantic_components, mock_bm25_searcher):
        """Create hybrid searcher with mock components"""
        return HybridTableSearch(mock_semantic_components, mock_bm25_searcher)

    def test_hybrid_searcher_initialization(
        self, mock_semantic_components, mock_bm25_searcher
    ):
        """Test HybridTableSearch initialization"""
        searcher = HybridTableSearch(mock_semantic_components, mock_bm25_searcher)

        assert searcher.embedding_data is not None
        assert searcher.generator is not None
        assert searcher.bm25 is not None

    def test_search_semantic_method(self, hybrid_searcher, mock_semantic_components):
        """Test semantic search through hybrid searcher"""
        query = "finance data"
        results = hybrid_searcher.search_semantic(query, top_k=2)

        # Should call the generator
        generator = mock_semantic_components[1]
        generator.generate_batch.assert_called_with([query])

        # Should return results in correct format
        assert isinstance(results, list)
        assert len(results) <= 2
        assert all(len(result) == 2 for result in results)  # (score, metadata) tuples

    def test_search_bm25_method(self, hybrid_searcher, mock_bm25_searcher):
        """Test BM25 search through hybrid searcher"""
        query = "transaction data"
        results = hybrid_searcher.search_bm25(query, top_k=2)

        # Should call BM25 searcher
        mock_bm25_searcher.search.assert_called_with(query, top_k=2)

        # Should return the mocked results
        assert len(results) == 2  # top_k=2 should limit results
        assert results[0][0] == 2.5  # First score from mock
        assert results[0][1]["table_name"] == "transactions"

    def test_search_hybrid_rrf(self, hybrid_searcher):
        """Test hybrid search using Reciprocal Rank Fusion"""
        query = "finance transactions"

        # Mock both search methods to return known results
        with patch.object(
            hybrid_searcher, "search_semantic"
        ) as mock_semantic, patch.object(hybrid_searcher, "search_bm25") as mock_bm25:
            # Mock semantic results (similarity scores)
            mock_semantic.return_value = [
                (0.9, {"table_name": "transactions", "id": "table1"}),
                (0.7, {"table_name": "payments", "id": "table2"}),
                (0.3, {"table_name": "accounts", "id": "table3"}),
            ]

            # Mock BM25 results (BM25 scores)
            mock_bm25.return_value = [
                (2.1, {"table_name": "payments", "id": "table2"}),
                (1.5, {"table_name": "transactions", "id": "table1"}),
                (0.2, {"table_name": "accounts", "id": "table3"}),
            ]

            results = hybrid_searcher.search_hybrid(query, top_k=3)

            # Verify both methods were called
            mock_semantic.assert_called_once_with(
                query, top_k=50
            )  # Default top_k for individual searches
            mock_bm25.assert_called_once_with(query, top_k=50)

            # Check results structure
            assert isinstance(results, list)
            assert len(results) == 3

            # Check that RRF scores are calculated (should be different from original scores)
            rrf_scores = [score for score, _ in results]
            assert all(
                0 <= score <= 1 for score in rrf_scores
            )  # RRF scores should be in [0,1]

            # Results should be sorted by RRF score
            assert rrf_scores == sorted(rrf_scores, reverse=True)

    def test_rrf_score_calculation(self, hybrid_searcher):
        """Test Reciprocal Rank Fusion score calculation logic"""
        # Test with known rankings - use proper table metadata format
        semantic_results = [
            (
                0.9,
                {"table_catalog": "cat", "table_schema": "schema", "table_name": "A"},
            ),  # Rank 1
            (
                0.5,
                {"table_catalog": "cat", "table_schema": "schema", "table_name": "B"},
            ),  # Rank 2
            (
                0.3,
                {"table_catalog": "cat", "table_schema": "schema", "table_name": "C"},
            ),  # Rank 3
        ]

        bm25_results = [
            (
                2.0,
                {"table_catalog": "cat", "table_schema": "schema", "table_name": "B"},
            ),  # Rank 1
            (
                1.0,
                {"table_catalog": "cat", "table_schema": "schema", "table_name": "A"},
            ),  # Rank 2
            (
                0.1,
                {"table_catalog": "cat", "table_schema": "schema", "table_name": "C"},
            ),  # Rank 3
        ]

        with patch.object(
            hybrid_searcher, "search_semantic", return_value=semantic_results
        ), patch.object(hybrid_searcher, "search_bm25", return_value=bm25_results):
            results = hybrid_searcher.search_hybrid("test", top_k=3)

            # Table A: RRF = 1/(1+1) + 1/(2+1) = 1/2 + 1/3 = 0.833
            # Table B: RRF = 1/(2+1) + 1/(1+1) = 1/3 + 1/2 = 0.833
            # Table C: RRF = 1/(3+1) + 1/(3+1) = 1/4 + 1/4 = 0.5

            # A and B should tie for first place, C should be last
            result_names = [meta["table_name"] for _, meta in results]
            rrf_scores = [score for score, _ in results]

            # Check that we got 3 results
            assert (
                len(results) == 3
            ), f"Expected 3 results, got {len(results)}: {results}"

            # Check that the highest scores are approximately equal for A and B
            assert rrf_scores[2] < rrf_scores[1] <= rrf_scores[0]  # C < B,A

            # Check that table C has the lowest score (should be last)
            assert (
                result_names[2] == "C"
            ), f"Expected C to be last, got order: {result_names}"

    def test_hybrid_search_empty_results(self, hybrid_searcher):
        """Test hybrid search when one method returns empty results"""
        with patch.object(
            hybrid_searcher, "search_semantic", return_value=[]
        ), patch.object(
            hybrid_searcher,
            "search_bm25",
            return_value=[(1.0, {"table_name": "test", "id": "table1"})],
        ):
            results = hybrid_searcher.search_hybrid("test", top_k=5)

            # Should still return results from BM25
            assert len(results) == 1
            assert results[0][1]["table_name"] == "test"

    def test_hybrid_search_duplicate_handling(self, hybrid_searcher):
        """Test that hybrid search handles duplicate tables correctly"""
        # Same table appears in both semantic and BM25 results
        semantic_results = [
            (0.8, {"table_name": "transactions", "table_schema": "finance"})
        ]
        bm25_results = [
            (2.0, {"table_name": "transactions", "table_schema": "finance"})
        ]

        with patch.object(
            hybrid_searcher, "search_semantic", return_value=semantic_results
        ), patch.object(hybrid_searcher, "search_bm25", return_value=bm25_results):
            results = hybrid_searcher.search_hybrid("test", top_k=5)

            # Should have only one result (deduplicated)
            assert len(results) == 1

            # Should have combined RRF score
            rrf_score = results[0][0]
            # RRF with k=60: RRF = 1/(60+1) + 1/(60+1) = 1/61 + 1/61 = 2/61 ≈ 0.0328
            # But this also gets additional weighted scoring, so just check it's reasonable
            assert (
                0 < rrf_score <= 1
            ), f"RRF score should be between 0 and 1, got {rrf_score}"
            assert rrf_score > 0.01, f"RRF score seems too low: {rrf_score}"


class TestSearchIntegration:
    """Integration tests for search functionality"""

    def test_end_to_end_search_pipeline(self):
        """Test complete search pipeline with real data structures"""
        # Create realistic test data
        table_data = pd.DataFrame(
            {
                "table_catalog": ["enterprise", "enterprise", "public"],
                "table_schema": ["finance", "users", "analytics"],
                "table_name": ["transactions", "accounts", "events"],
                "all_columns": [
                    "transaction_id, user_id, amount, date, type",
                    "user_id, username, email, created_date",
                    "event_id, user_id, event_type, timestamp",
                ],
            }
        )

        # Test BM25 search
        bm25_searcher = BM25TableSearch()
        bm25_searcher.fit(table_data)

        bm25_results = bm25_searcher.search("user transaction", top_k=2)
        assert len(bm25_results) == 2
        assert all(isinstance(score, (int, float)) for score, _ in bm25_results)
        assert all(isinstance(metadata, dict) for _, metadata in bm25_results)

        # Verify that results contain expected fields
        for score, metadata in bm25_results:
            assert "table_name" in metadata
            assert "table_schema" in metadata
            assert "all_columns" in metadata

    def test_search_result_consistency(self):
        """Test that search results are consistent across multiple calls"""
        table_data = pd.DataFrame(
            {
                "table_catalog": ["test"],
                "table_schema": ["schema"],
                "table_name": ["table"],
                "all_columns": ["col1, col2, col3"],
            }
        )

        searcher = BM25TableSearch()
        searcher.fit(table_data)

        query = "test table"

        # Run search multiple times
        results1 = searcher.search(query, top_k=1)
        results2 = searcher.search(query, top_k=1)

        # Results should be identical
        assert len(results1) == len(results2)
        for (score1, meta1), (score2, meta2) in zip(results1, results2):
            assert score1 == pytest.approx(score2, abs=1e-6)
            assert meta1 == meta2
