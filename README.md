# BigQuery Data Dictionary Search

A comprehensive search system for BigQuery table metadata that combines semantic similarity search with keyword-based retrieval to help users discover relevant tables in large datasets.

## Features

- **Multi-Modal Search**: Combines semantic similarity (using sentence transformers) with BM25 keyword matching
- **BigQuery Integration**: Direct integration with Google Cloud BigQuery to retrieve table metadata
- **Interactive UIs**: Both Streamlit and FastAPI interfaces for different use cases
- **Comprehensive Evaluation**: Built-in evaluation system with metrics like Precision@K, Recall@K, NDCG, and MRR
- **Flexible Data Sources**: Use your own BigQuery datasets or provided example data

## Architecture

The system consists of several key components:

1. **Data Retrieval** (`src/retrieve_data.py`): Fetches table metadata from BigQuery
2. **Search Methods**:
   - **Semantic Search** (`src/semantic/`): Uses sentence transformers for embedding-based similarity
   - **BM25 Search** (`src/bm25/`): Traditional keyword-based search using BM25 algorithm
   - **Hybrid Search** (`src/hybrid/`): Combines both methods using Reciprocal Rank Fusion (RRF)
3. **Model Preparation** (`src/prepare_models.py`): Trains and saves embedding models and BM25 indices
4. **Evaluation System** (`src/evaluate.py`): Comprehensive evaluation with multiple metrics
5. **User Interfaces**:
   - **Streamlit App** (`src/streamlit_app.py`): Interactive web interface
   - **FastAPI App** (`src/fastapi_app.py`): REST API for programmatic access

## Installation

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- Google Cloud credentials (for BigQuery access)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd bigquery-data-dictionary
```

2. Install dependencies:
```bash
uv sync
```

3. (Optional) Set up Google Cloud authentication:
```bash
gcloud auth application-default login
```

## Quick Start

### Using Example Data

To get started quickly with example data:

```bash
./setup.sh
```

This will:
1. Install dependencies
2. Use the provided example dataset
3. Train semantic embeddings and BM25 models
4. Run evaluation benchmarks

### Using Your Own BigQuery Data

To use your own BigQuery datasets:

```bash
./setup.sh YOUR_PROJECT_ID TARGET_PROJECT_ID DATASET_ID
```

Example:
```bash
./setup.sh my-project bigquery-public-data austin_311
```

### Starting the Applications

**Streamlit Interface (Default):**
```bash
./start_app.sh
```

**FastAPI Interface:**
```bash
./start_app.sh fastapi
```

## Usage

### Streamlit Interface

The Streamlit app provides an intuitive web interface where you can:

- Enter natural language queries to search for tables
- Choose between different search methods (Semantic, BM25, or Hybrid)
- View search results with relevance scores
- Explore table metadata including column information
- Try example queries to understand the system capabilities

### FastAPI Interface

The FastAPI provides REST endpoints for programmatic access:

- `POST /search`: Search for tables using different methods
- `GET /health`: Health check endpoint

### Search Methods

1. **Semantic Search**: Uses sentence transformers to find tables based on meaning and context
2. **BM25 Search**: Traditional keyword matching with TF-IDF-like scoring
3. **Hybrid Search**: Combines both methods using Reciprocal Rank Fusion for optimal results

## Data Format

The system expects table metadata in pipe-separated CSV format:

```
table_catalog|table_schema|table_name|all_columns
enterprise|telecom_core|customer_accounts|customer_id, account_number, first_name, last_name...
```

## Evaluation

The system includes comprehensive evaluation capabilities:

```bash
uv run src/evaluate.py --test-dataset path/to/test/dataset.csv
```

Evaluation metrics include:
- **Precision@K**: Fraction of relevant results in top K
- **Recall@K**: Fraction of relevant results retrieved
- **Mean Reciprocal Rank (MRR)**: Average reciprocal rank of first relevant result
- **NDCG**: Normalized Discounted Cumulative Gain
- **F1 Score**: Harmonic mean of precision and recall

## Project Structure

```
├── src/
│   ├── semantic/           # Semantic search implementation
│   ├── bm25/              # BM25 keyword search
│   ├── hybrid/            # Hybrid search combining both
│   ├── retrieve_data.py   # BigQuery data retrieval
│   ├── prepare_models.py  # Model training and preparation
│   ├── evaluate.py        # Evaluation system
│   ├── streamlit_app.py   # Streamlit web interface
│   └── fastapi_app.py     # FastAPI REST interface
├── data/
│   ├── raw/              # Raw datasets
│   └── embedding/        # Saved embeddings
├── models/               # Trained models (BM25, embeddings)
├── evals/               # Evaluation results
├── setup.sh             # Setup and training pipeline
└── start_app.sh         # Application launcher
```

## Configuration

### Model Configuration

- **Semantic Model**: Uses `all-MiniLM-L6-v2` sentence transformer by default
- **BM25 Parameters**: Standard BM25 with k1=1.2, b=0.75
- **Hybrid Fusion**: Reciprocal Rank Fusion with equal weighting

## Troubleshooting

### Common Issues

1. **Authentication Error**: Ensure Google Cloud credentials are properly configured
2. **Model Loading Error**: Check that models have been trained using `setup.sh`
3. **Memory Issues**: For large datasets, consider increasing system memory or reducing batch sizes

### Getting Help

- Check the evaluation results in the `evals/` directory for performance insights
- Review the example queries in the Streamlit interface
- Examine the test dataset format in `data/raw/dataset_example_test.csv`
