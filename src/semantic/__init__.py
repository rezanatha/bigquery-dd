import pickle
from typing import List
import numpy as np


class EmbeddingGenerator:
    def __init__(self, model=None):
        """Initialize with model loaded once"""
        self.model = model

    def generate_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for multiple texts efficiently"""
        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(
            texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True
        )
        return embeddings

    def save_model(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    def load_model(self, filepath):
        with open(filepath, "rb") as f:
            generator = pickle.load(f)
            self.model = generator.model
