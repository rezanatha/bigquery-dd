#!/bin/bash

## install uv packages (uv sync)
echo "Step 1: UV sync"
uv sync;

# run retrieve_data.py or just use example_dataset (default)
if [ $# -eq 0 ]; then
    echo "Step 2: No arguments provided, using example_dataset.csv"
    DATASET_FILE="data/raw/dataset_example.csv"
else
    echo "Step 2: Retrieve data"
    uv run python src/retrieve_data.py $1 --target-project $2 --dataset $3
    DATASET_FILE="data/raw/retrieved_output.csv"
fi

# run prepare_models.py
echo "Step 3: Prepare Semantic LM embedding and BM25 model"
uv run src/prepare_models.py --dataset "$DATASET_FILE"

# run evaluate.py
echo "Step 4: Evaluate embedding and model"
DATASET_TEST_FILE="data/raw/dataset_example_test.csv"
uv run src/evaluate.py --test-dataset "$DATASET_TEST_FILE"
