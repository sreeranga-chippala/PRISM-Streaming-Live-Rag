"""
Streaming Live RAG package initialization.

The macOS ML stack used by FAISS, PyTorch, NumPy, and
SentenceTransformers can otherwise create a native threading
interaction that causes a segmentation fault during retrieval.
"""

import os
import sys


if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")