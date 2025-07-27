from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
import uvicorn
from semantic import semantic_search, EmbeddingGenerator  # noqa: F401
import pandas as pd

# Initialize FastAPI app
app = FastAPI(
    title="BigQuery Data Dictionary Search API",
    description="Search through table metadata using semantic similarity",
    version="1.0.0",
)

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for search components
embedding_data = None
generator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load search components on startup"""
    global embedding_data, generator
    try:
        embedding_data, generator = semantic_search.load_search_components()
        print("✅ Search components loaded successfully")
        yield
    except Exception as e:
        print(f"❌ Failed to load search components: {e}")
        raise e


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="BigQuery Data Dictionary Search API",
    description="Search through table metadata using semantic similarity",
    version="1.0.0",
    lifespan=lifespan,
)


# Pydantic models for request/response
class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10


class TableResult(BaseModel):
    rank: int
    similarity: float
    table_catalog: str
    table_schema: str
    table_name: str
    all_columns: str
    quality_level: str


class SearchResponse(BaseModel):
    query: str
    total_results: int
    results: List[TableResult]
    statistics: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    message: str


@app.on_event("startup")
async def startup_event():
    """Load search components on startup"""
    global embedding_data, generator
    try:
        embedding_data, generator = semantic_search.load_search_components()
        print("✅ Search components loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load search components: {e}")
        raise e


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if embedding_data is None or generator is None:
        raise HTTPException(status_code=503, detail="Search components not loaded")

    return HealthResponse(status="healthy", message="Search API is ready")


@app.post("/search", response_model=SearchResponse)
async def search_tables(request: SearchRequest):
    """Search for similar tables"""
    if embedding_data is None or generator is None:
        raise HTTPException(status_code=503, detail="Search components not loaded")

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        # Perform search
        results = semantic_search.search_similar_tables(
            request.query, embedding_data, generator, top_k=request.top_k
        )

        # Format results
        formatted_results = []
        for i, (score, metadata) in enumerate(results, 1):
            # Determine quality level
            if score >= 0.6:
                quality_level = "High"
            elif score >= 0.4:
                quality_level = "Medium"
            else:
                quality_level = "Low"

            formatted_results.append(
                TableResult(
                    rank=i,
                    similarity=float(score),  # Convert numpy.float32 to Python float
                    table_catalog=metadata.get("table_catalog", "N/A"),
                    table_schema=metadata.get("table_schema", "N/A"),
                    table_name=metadata.get("table_name", "N/A"),
                    all_columns=metadata.get("all_columns", "N/A"),
                    quality_level=quality_level,
                )
            )

        # Calculate statistics
        scores = [score for score, _ in results]
        schemas = [metadata.get("table_schema", "Unknown") for _, metadata in results]
        schema_counts = pd.Series(schemas).value_counts().to_dict()

        high_quality = sum(1 for score, _ in results if score >= 0.6)
        medium_quality = sum(1 for score, _ in results if 0.4 <= score < 0.6)
        low_quality = sum(1 for score, _ in results if score < 0.4)

        statistics = {
            "best_match": float(max(scores)) if scores else 0.0,
            "worst_match": float(min(scores)) if scores else 0.0,
            "average_score": float(sum(scores) / len(scores)) if scores else 0.0,
            "score_range": float(max(scores) - min(scores)) if scores else 0.0,
            "schema_distribution": schema_counts,
            "quality_distribution": {
                "high_quality_count": high_quality,
                "medium_quality_count": medium_quality,
                "low_quality_count": low_quality,
                "high_quality_ratio": float(high_quality / len(results))
                if results
                else 0.0,
            },
        }

        return SearchResponse(
            query=request.query,
            total_results=len(formatted_results),
            results=formatted_results,
            statistics=statistics,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/example-queries")
async def get_example_queries():
    """Get example search queries"""
    return {
        "examples": [
            "fraud detection tables",
            "customer analytics",
            "network performance monitoring",
            "billing and payments",
            "user authentication",
            "data warehouse facts",
            "staging tables",
            "compliance and audit",
        ]
    }


if __name__ == "__main__":
    uvicorn.run(
        "fastapi_app:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )
