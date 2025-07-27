import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List
import pickle
from components import EmbeddingGenerator


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


def load_dataset(data_path: pd.DataFrame):
    data = pd.read_csv(data_path, sep="|").sample(frac=1).reset_index(drop=True)
    data["text_form"] = data.apply(create_text, axis=1)
    return data


def generate_embedding(data, generator):
    texts = data["text_form"].tolist()
    metadata = data[
        ["table_catalog", "table_schema", "table_name", "all_columns"]
    ].to_dict("records")

    # Generate embeddings
    embeddings = generator.generate_batch(texts, batch_size=32)

    # Add text to metadata for reference
    for i, meta in enumerate(metadata):
        meta["search_text"] = texts[i]
        meta["embedding_index"] = i

    return embeddings, metadata


def save_embeddings(
    embeddings: np.ndarray, metadata: List[dict], model_name: str, filepath: str
):
    """Save embeddings with metadata"""
    data = {
        "embeddings": embeddings,
        "metadata": metadata,
        "model_name": model_name,
        "embedding_dim": embeddings.shape[1],
    }
    with open(filepath, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved {len(embeddings)} embeddings to {filepath}")


if __name__ == "__main__":
    data = load_dataset("data/raw/dataset_example.csv")

    # generate and save embedding
    model_name = "all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)

    generator = EmbeddingGenerator(model)
    embeddings, metadata = generate_embedding(data, generator)
    save_embeddings(
        embeddings, metadata, model_name, "data/embedding/miniLM_embedding.pkl"
    )
    generator.save_model("models/miniLM_embedding_generator.pkl")
    print(embeddings.shape)
