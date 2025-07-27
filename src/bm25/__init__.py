from typing import List, Dict, Tuple
import pandas as pd
from rank_bm25 import BM25Okapi


class BM25TableSearch:
    def __init__(self):
        self.bm25 = None
        self.metadata = []

    def _preprocess_text(self, text: str) -> str:
        if not text:
            return ""

        # Convert to lowercase
        text = text.lower()

        # Handle database naming conventions
        text = text.replace("_", " ")  # user_id -> user id
        text = text.replace(".", " ")  # schema.table -> schema table
        text = text.replace("tbl", "table")
        text = text.replace("dim_", "dimension ")
        text = text.replace("fact_", "fact ")
        text = text.replace("stg_", "staging ")

        # Expand common abbreviations
        replacements = {
            " id ": " identifier ",
            " desc ": " description ",
            " mgmt ": " management ",
            " perf ": " performance ",
            " auth ": " authentication ",
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        return text

    def _create_document(self, row: Dict) -> str:
        """Create searchable document from table row"""
        table_name = row.get("table_name", "")
        table_schema = row.get("table_schema", "")
        columns = row.get("all_columns", "")

        # Weight important fields by repetition
        parts = []

        # Table name is most important (repeat 3x)
        if table_name:
            parts.extend([table_name] * 3)

        # Schema is important (repeat 2x)
        if table_schema:
            parts.extend([table_schema] * 2)

        # Columns get normal weight
        if columns:
            parts.append(columns)

        doc = " ".join(parts)
        return self._preprocess_text(doc)

    def fit(self, data: pd.DataFrame):
        """Build BM25 index from dataframe"""
        print("Building BM25 index with rank-bm25...")

        # Prepare documents and metadata
        documents = []
        self.metadata = []

        for _, row in data.iterrows():
            # Create document text
            doc_text = self._create_document(row.to_dict())

            # Tokenize (simple split for BM25Okapi)
            doc_tokens = doc_text.split()

            documents.append(doc_tokens)
            self.metadata.append(
                {
                    "table_catalog": row.get("table_catalog", ""),
                    "table_schema": row.get("table_schema", ""),
                    "table_name": row.get("table_name", ""),
                    "all_columns": row.get("all_columns", ""),
                }
            )

        # Build BM25 index
        self.bm25 = BM25Okapi(documents)

        print(f"Built BM25 index for {len(documents)} tables")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[float, Dict]]:
        """Search for tables using BM25"""
        if self.bm25 is None:
            return []

        # Preprocess and tokenize query
        processed_query = self._preprocess_text(query)
        query_tokens = processed_query.split()

        if not query_tokens:
            return []

        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)

        # Create results with metadata
        results = []
        for i, score in enumerate(scores):
            if score > 0:  # Only include non-zero scores
                results.append((float(score), self.metadata[i]))

        # Sort by score and return top k
        results.sort(key=lambda x: x[0], reverse=True)
        return results[:top_k]
