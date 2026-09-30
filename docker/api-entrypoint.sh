#!/bin/sh
set -eu

VECTOR_DIR="data/processed/vector_store"

if [ ! -d "$VECTOR_DIR" ] || [ -z "$(find "$VECTOR_DIR" -type f -print -quit 2>/dev/null)" ]; then
  echo "[PRISM] Vector store not found. Building it from committed PDFs..."
  python scripts/ingest_documents.py
  python scripts/build_index.py
else
  echo "[PRISM] Existing vector store found. Skipping ingestion."
fi

exec "$@"
