import streamlit as st
import pandas as pd
from semantic import semantic_search, EmbeddingGenerator  # noqa: F401
from bm25 import bm25_search, BM25TableSearch  # noqa: F401
from hybrid import HybridTableSearch

# Configure page
st.set_page_config(
    page_title="BigQuery Data Dictionary Search", page_icon="🔍", layout="wide"
)


@st.cache_resource
def load_search_components_for_streamlit():
    """Load hybrid search components (cached for performance)"""
    try:
        # Load semantic search components
        semantic_component = semantic_search.load_search_components()

        # Load BM25 search
        bm25_component = bm25_search.load_search_components()

        # Initialize hybrid search
        hybrid_searcher = HybridTableSearch(semantic_component, bm25_component)

        return hybrid_searcher

    except Exception as e:
        st.error(f"Error loading search components: {e}")
        return None


def main():
    st.title("🔍 BigQuery Data Dictionary Search")
    st.markdown(
        "Search through table metadata using semantic similarity and keyword matching"
    )

    # Initialize session state
    if "search_query" not in st.session_state:
        st.session_state.search_query = ""

    # Load hybrid search components
    hybrid_searcher = load_search_components_for_streamlit()

    if hybrid_searcher is None:
        st.error(
            "Failed to load hybrid search components. Please check your model files."
        )
        return

    # Search interface
    col1, col2 = st.columns([3, 1])

    with col1:
        query = st.text_input(
            "Search for tables:",
            value=st.session_state.search_query,
            placeholder=(
                "e.g., 'fraud detection tables', 'customer analytics', "
                "'network performance'..."
            ),
            help="Enter keywords to find relevant tables",
        )

    with col2:
        top_k = st.number_input(
            "Number of results:", min_value=1, max_value=50, value=10
        )

    # Search options
    with st.expander("🔧 Search Options", expanded=False):
        search_method = st.selectbox(
            "Search Method:",
            ["Hybrid (Semantic + Keyword)", "Semantic Only", "Keyword Only"],
            index=0,
        )

    # Search button and results
    search_triggered = st.button("🔍 Search", type="primary")

    if search_triggered and query:
        if query.strip():
            with st.spinner("Searching..."):
                if search_method == "Hybrid (Semantic + Keyword)":
                    results = hybrid_searcher.search_hybrid(query, top_k=top_k)
                elif search_method == "Semantic Only":
                    results = hybrid_searcher.search_semantic(query, top_k=top_k)
                else:  # Keyword Only
                    results = hybrid_searcher.search_bm25(query, top_k=top_k)

            if results:
                st.success(f"Found {len(results)} results for: '{query}'")

                # Display results
                for i, (score, metadata) in enumerate(results, 1):
                    with st.container():
                        col1, col2, col3 = st.columns([0.5, 2, 1])

                        with col1:
                            st.metric("Rank", f"#{i}")

                        with col2:
                            table_name = (
                                f"{metadata.get('table_catalog', 'N/A')}."
                                f"{metadata.get('table_schema', 'N/A')}."
                                f"{metadata.get('table_name', 'N/A')}"
                            )
                            st.subheader(table_name)

                            # Show columns in expandable section
                            columns = metadata.get("all_columns", "N/A")
                            with st.expander("Show columns"):
                                st.write(columns)

                        with col3:
                            # Show raw score without arbitrary categorization
                            score_label = (
                                "Similarity"
                                if search_method == "Semantic Only"
                                else "Score"
                            )
                            if search_method == "Semantic Only":
                                st.metric(score_label, f"{score:.3f}")
                            else:
                                st.metric(score_label, f"{score:.4f}")

                        st.divider()

                # Summary statistics
                with st.sidebar:
                    st.header("Search Statistics")
                    scores = [score for score, _ in results]

                    st.metric("Best Match", f"{max(scores):.3f}")
                    st.metric("Average Score", f"{sum(scores)/len(scores):.3f}")
                    st.metric("Score Range", f"{max(scores) - min(scores):.3f}")

                    # Schema breakdown
                    schemas = [
                        metadata.get("table_schema", "Unknown")
                        for _, metadata in results
                    ]
                    schema_counts = pd.Series(schemas).value_counts()

                    st.subheader("Schemas Found")
                    for schema, count in schema_counts.items():
                        st.write(f"• {schema}: {count}")

            else:
                st.warning("No results found. Try different keywords.")
        else:
            st.info("Please enter a search query.")

    # Example queries
    with st.sidebar:
        st.header("Example Queries")
        example_queries = [
            "fraud detection tables",
            "customer analytics",
            "network performance monitoring",
            "billing and payments",
            "user authentication",
            "data warehouse facts",
            "staging tables",
            "compliance and audit",
        ]

        for example in example_queries:
            if st.button(f"📋 {example}", key=f"example_{example}"):
                st.session_state.search_query = example
                st.rerun()

    # Footer
    st.markdown("---")


if __name__ == "__main__":
    main()
