#!/bin/bash

# run streamlit or fastapi (default streamlit)

if [ "$1" = "fastapi" ]; then
    echo "Starting FastAPI application..."
    uv run src/fastapi_app.py
else
    echo "Starting Streamlit application..."
    uv run streamlit run src/streamlit_app.py
fi
