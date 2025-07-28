# BigQuery Data Dictionary Search

A comprehensive search system for BigQuery table metadata that combines semantic similarity search with keyword-based retrieval to help users discover relevant tables in BigQuery.

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

1. Go to the repository:
```bash
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

The system includes comprehensive evaluation capabilities with a custom relevance scoring system:

```bash
uv run src/evaluate.py --test-dataset path/to/test/dataset.csv
```

### Custom Relevance Scoring

The evaluation uses a keyword-based relevance scoring system that rewards tables with more matching keywords:

- **0 points**: No keyword matches between query and table
- **1.0 points**: Base score for any keyword match
- **+0.5 points**: For each additional matching keyword

**Example:**
- Query: `["fraud", "detection", "security"]`
- Table keywords: `["fraud", "detection", "risk", "investigation"]`
- Matching keywords: `["fraud", "detection"]` (2 matches)
- **Relevance Score**: 1.0 + (2-1) × 0.5 = **1.5**

This scoring approach ensures that tables with more comprehensive keyword coverage receive higher relevance scores, providing more nuanced evaluation than binary relevant/non-relevant judgments.

### Test Dataset Format

The test dataset includes a `relevant_keywords` column for ground truth:

```
table_catalog|table_schema|table_name|all_columns|relevant_keywords
enterprise|financial_services|fraud_detection|fraud_id, customer_id...|fraud, detection, security, risk
```

### Evaluation Metrics

The custom relevance scores integrate with standard information retrieval metrics:

#### Binary-Based Metrics
- **Precision@K**: Counts tables with relevance > 0 as relevant (1.0, 1.5, 2.0 all count as "relevant")
- **Recall@K**: Same binary treatment - any positive relevance score counts
- **Mean Reciprocal Rank (MRR)**: Position of first table with relevance > 0

#### Graded Relevance Metrics
- **Normalized Discounted Cumulative Gain (NDCG)**: Fully utilizes the graded scores! Higher relevance scores (2.0 vs 1.0) contribute more to DCG
  - **DCG formula**: `Σ(relevance_score / log2(rank + 1))` for all retrieved results
  - **IDCG formula**: `Σ(relevance_score / log2(rank + 1))` for perfect ranking (highest scores first)
  - **NDCG = DCG / IDCG** (normalized to 0-1 scale)
  - Example: Table with score 2.0 at rank 1 contributes 2.0/log2(2) = 2.0 to DCG
  - NDCG accounts for the ideal ranking to provide comparable scores across different queries
- **Average Relevance**: Mean of actual relevance scores for retrieved relevant tables

#### Example Impact
For query results: `[Table A (score=2.0), Table B (score=1.5), Table C (score=0)]`

- **Precision@10**: 2/3 = 0.67 (binary: A and B both count as relevant)
- **NDCG**: Accounts for graded relevance - Table A contributes more than Table B
- **Average Relevance**: (2.0 + 1.5) / 2 = 1.75

This approach rewards search systems that not only find relevant tables but rank the most comprehensive matches (higher keyword overlap) at the top.

The system evaluates all three search methods (semantic, BM25, hybrid) and saves detailed results to the `evals/` directory for comparison.

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
- **Hybrid Fusion**: [Reciprocal Rank Fusion](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion) with equal weighting

## Testing

### Running Tests

The project includes comprehensive unit tests for all major components:

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage report
uv run pytest tests/ --cov=src --cov-report=html

# Run specific test files
uv run pytest tests/test_embedding.py -v
uv run pytest tests/test_search.py -v
```

### Test Coverage

- **test_embedding.py**: Tests embedding generation, text processing, and model serialization
- **test_search.py**: Tests semantic search, BM25 search, and hybrid fusion algorithms
- **Integration tests**: End-to-end pipeline testing with real data structures

### Code Quality

The project uses pre-commit hooks for code quality:

```bash
# Install pre-commit hooks
uv run pre-commit install

# Run all hooks manually
uv run pre-commit run --all-files
```

**Quality tools**:
- **black**: Code formatting
- **flake8**: Linting and style checking
- **autoflake**: Remove unused imports
- **trailing-whitespace**: File cleanup

## Continuous Integration

### GitHub Actions

The project uses GitHub Actions for automated testing:

- **Triggers**: Runs on every push/PR to `main` and `develop` branches
- **Python Version**: Tests against Python 3.11
- **Quality Gates**: All PRs must pass linting and tests before merging
- **Coverage Reporting**: Automatic upload to Codecov

### CI Pipeline Steps

1. **Setup**: Install Python 3.11 and uv package manager
2. **Dependencies**: Install project dependencies with `uv sync`
3. **Code Quality**: Run pre-commit hooks (linting, formatting)
4. **Testing**: Execute full test suite with coverage reporting
5. **Integration**: Test the setup script with example data

### Branch Protection

- **main branch**: Protected, requires PR reviews and passing CI
- **develop branch**: Integration branch for feature development
- **Feature branches**: Use `feature/description` naming convention

## Troubleshooting

### Common Issues

1. **Authentication Error**: Ensure Google Cloud credentials are properly configured
2. **Model Loading Error**: Check that models have been trained using `setup.sh`
3. **Memory Issues**: For large datasets, consider increasing system memory or reducing batch sizes
4. **Test Failures**: Run `uv run pytest tests/ -v` to see detailed test output
5. **CI Failures**: Check GitHub Actions logs for specific error messages

### Getting Help

- Check the evaluation results in the `evals/` directory for performance insights
- Review the example queries in the Streamlit interface
- Examine the test dataset format in `data/raw/dataset_example_test.csv`
- View test coverage reports in `htmlcov/index.html` after running tests with coverage
