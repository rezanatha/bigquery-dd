import pandas as pd
from semantic import (
    embedding,
    EmbeddingGenerator,
)
from bm25 import bm25_search, BM25TableSearch


# CREATE EMBEDDING FOR SEMANTIC MODEL
def create_embedding(dataset_path: str):
    data = embedding.load_dataset(dataset_path)

    # generate and save embedding
    model_name = "all-MiniLM-L6-v2"
    model = embedding.SentenceTransformer(model_name)

    generator = EmbeddingGenerator(model)
    embeddings, metadata = embedding.generate_embedding(data, generator)
    embedding.save_embeddings(
        embeddings, metadata, model_name, "data/embedding/miniLM_embedding.pkl"
    )
    generator.save_model("models/miniLM_embedding_generator.pkl")
    print(embeddings.shape)


# TRAIN BM25
def train_bm25(dataset_path: str):
    data = pd.read_csv(dataset_path, sep="|")
    bm25 = BM25TableSearch()
    bm25.fit(data)
    bm25_search.save_bm25_model(bm25, "models/bm25.pkl")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare semantic embedding and BM25 models"
    )
    parser.add_argument(
        "--dataset",
        default="data/raw/dataset_example.csv",
        help="Path to dataset CSV file (default: data/raw/dataset_example.csv)",
    )

    args = parser.parse_args()

    create_embedding(args.dataset)
    train_bm25(args.dataset)
