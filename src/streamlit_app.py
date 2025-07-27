import streamlit as st
import pandas as pd
from search import load_search_components, search_similar_tables


# Configure page
st.set_page_config(
    page_title="BigQuery Data Dictionary Search",
    page_icon="🔍",
    layout="wide"
)

@st.cache_resource
def load_search_components_for_streamlit():
    """Load embeddings and model (cached for performance)"""
    try:
        embedding_data, generator = load_search_components()
        return embedding_data, generator
    except Exception as e:
        st.error(f"Error loading search components: {e}")
        return None, None

def main():
    st.title("🔍 BigQuery Data Dictionary Search")
    st.markdown("Search through table metadata using semantic similarity")

    # Initialize session state
    if 'search_query' not in st.session_state:
        st.session_state.search_query = ""

    # Load components
    embedding_data, generator = load_search_components()

    if embedding_data is None or generator is None:
        st.error("Failed to load search components. Please check your model files.")
        return

    # Search interface
    col1, col2 = st.columns([3, 1])

    with col1:
        query = st.text_input(
            "Search for tables:",
            value=st.session_state.search_query,
            placeholder="e.g., 'fraud detection tables', 'customer analytics', 'network performance'...",
            help="Enter keywords to find relevant tables"
        )

    with col2:
        top_k = st.number_input("Number of results:", min_value=1, max_value=50, value=10)

    # Search button and results
    search_triggered = st.button("🔍 Search", type="primary")

    if search_triggered and query:
        if query.strip():
            with st.spinner("Searching..."):
                results = search_similar_tables(query, embedding_data, generator, top_k=top_k)

            if results:
                st.success(f"Found {len(results)} results for: '{query}'")

                # Display results
                for i, (score, metadata) in enumerate(results, 1):
                    with st.container():
                        col1, col2, col3 = st.columns([0.5, 2, 1])

                        with col1:
                            st.metric("Rank", f"#{i}")

                        with col2:
                            table_name = f"{metadata.get('table_catalog', 'N/A')}.{metadata.get('table_schema', 'N/A')}.{metadata.get('table_name', 'N/A')}"
                            st.subheader(table_name)

                            # Show columns in expandable section
                            columns = metadata.get('all_columns', 'N/A')
                            with st.expander("Show columns"):
                                st.write(columns)

                        with col3:
                            # Similarity score with color coding
                            if score >= 0.6:
                                st.metric("Similarity", f"{score:.3f}", delta="High", delta_color="normal")
                            elif score >= 0.4:
                                st.metric("Similarity", f"{score:.3f}", delta="Medium", delta_color="off")
                            else:
                                st.metric("Similarity", f"{score:.3f}", delta="Low", delta_color="inverse")

                        st.divider()

                # Summary statistics
                with st.sidebar:
                    st.header("Search Statistics")
                    scores = [score for score, _ in results]

                    st.metric("Best Match", f"{max(scores):.3f}")
                    st.metric("Average Score", f"{sum(scores)/len(scores):.3f}")
                    st.metric("Score Range", f"{max(scores) - min(scores):.3f}")

                    # Schema breakdown
                    schemas = [metadata.get('table_schema', 'Unknown') for _, metadata in results]
                    schema_counts = pd.Series(schemas).value_counts()

                    st.subheader("Schemas Found")
                    for schema, count in schema_counts.items():
                        st.write(f"• {schema}: {count}")

                    # Quality distribution
                    high_quality = sum(1 for score, _ in results if score >= 0.6)
                    medium_quality = sum(1 for score, _ in results if 0.4 <= score < 0.6)
                    low_quality = sum(1 for score, _ in results if score < 0.4)

                    st.subheader("Quality Distribution")
                    st.write(f"🟢 High (≥0.6): {high_quality}")
                    st.write(f"🟡 Medium (0.4-0.6): {medium_quality}")
                    st.write(f"🔴 Low (<0.4): {low_quality}")

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
            "compliance and audit"
        ]

        for example in example_queries:
            if st.button(f"📋 {example}", key=f"example_{example}"):
                st.session_state.search_query = example
                st.rerun()

    # Footer
    st.markdown("---")
if __name__ == "__main__":
    main()
