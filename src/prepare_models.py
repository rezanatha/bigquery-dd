import pandas as pd
from semantic import (
    embedding,
    EmbeddingGenerator,
)
from bm25 import bm25_search, BM25TableSearch


# CREATE EMBEDDING FOR SEMANTIC MODEL
def create_embedding():
    data = embedding.load_dataset("data/raw/dataset_example.csv")

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
def train_bm25():
    data = pd.read_csv("data/raw/dataset_example.csv", sep="|")
    bm25 = BM25TableSearch()
    bm25.fit(data)
    bm25_search.save_bm25_model(bm25, "models/bm25.pkl")


if __name__ == "__main__":
    create_embedding()
    train_bm25()
