from pathlib import Path
import json
import traceback

from src.retrieval.document_loader import DocumentLoader
from src.retrieval.chunker import TextChunker
from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.vector_store import VectorStore


# ============================================================
# CONFIGURATION
# ============================================================

RAW_DIR = Path(
    "data/raw/documents"
)

CHUNKS_DIR = Path(
    "data/processed/chunks"
)

VECTOR_STORE_DIR = Path(
    "data/processed/vector_store"
)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

EMBEDDING_MODEL_NAME = (
    "all-MiniLM-L6-v2"
)

EMBEDDING_BATCH_SIZE = 32


# ============================================================
# MAIN
# ============================================================

def build_index():

    print("=" * 70)
    print("BUILD VECTOR INDEX")
    print("=" * 70)

    # --------------------------------------------------------
    # CREATE DIRECTORIES
    # --------------------------------------------------------

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    CHUNKS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    VECTOR_STORE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # FIND DOCUMENTS
    # --------------------------------------------------------

    files = [
        path
        for path in RAW_DIR.rglob("*")
        if path.is_file()
        and not path.name.startswith(".")
    ]

    if not files:

        print()
        print(
            f"No documents found in: {RAW_DIR}"
        )

        print()
        print(
            "Put your documents inside:"
        )

        print(
            f"  {RAW_DIR}"
        )

        return

    print()
    print(
        f"Input directory : {RAW_DIR}"
    )

    print(
        f"Chunks directory: {CHUNKS_DIR}"
    )

    print(
        f"Vector store    : {VECTOR_STORE_DIR}"
    )

    print(
        f"Files found     : {len(files)}"
    )

    # --------------------------------------------------------
    # INITIALIZE COMPONENTS
    # --------------------------------------------------------

    print()
    print(
        "Initializing document loader..."
    )

    loader = DocumentLoader()

    print(
        "Initializing chunker..."
    )

    chunker = TextChunker(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    print(
        "Initializing embedding model..."
    )

    embedding_model = EmbeddingModel(
        model_name=EMBEDDING_MODEL_NAME,
        batch_size=EMBEDDING_BATCH_SIZE,
    )

    print(
        "Initializing vector store..."
    )

    vector_store = VectorStore(
        index_dir=VECTOR_STORE_DIR,
        dimension=embedding_model.get_dimension(),
    )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    total_files = 0
    successful_files = 0
    failed_files = 0

    total_documents = 0
    total_chunks = 0
    total_vectors = 0

    errors = []

    # --------------------------------------------------------
    # PROCESS DOCUMENTS
    # --------------------------------------------------------

    for file_path in files:

        total_files += 1

        print()
        print("-" * 70)
        print(
            f"Processing: {file_path}"
        )
        print("-" * 70)

        file_documents = 0
        file_chunks = 0
        file_vectors = 0

        try:

            # ------------------------------------------------
            # LOAD DOCUMENT
            # ------------------------------------------------

            documents = loader.iter_load(
                file_path
            )

            # ------------------------------------------------
            # PROCESS EACH DOCUMENT
            # ------------------------------------------------

            for document in documents:

                file_documents += 1
                total_documents += 1

                # --------------------------------------------
                # CHUNK DOCUMENT
                # --------------------------------------------

                chunks = list(
                    chunker.chunk_document(
                        document
                    )
                )

                if not chunks:
                    continue

                # --------------------------------------------
                # SAVE CHUNKS
                # --------------------------------------------

                save_chunks(
                    chunks,
                    CHUNKS_DIR
                )

                file_chunks += len(chunks)
                total_chunks += len(chunks)

                # --------------------------------------------
                # CREATE EMBEDDINGS
                # --------------------------------------------

                texts = [
                    chunk.text
                    for chunk in chunks
                ]

                embeddings = (
                    embedding_model.encode_batch(
                        texts
                    )
                )

                # --------------------------------------------
                # ADD TO VECTOR STORE
                # --------------------------------------------

                added = vector_store.add(
                    embeddings,
                    chunks
                )

                file_vectors += added
                total_vectors += added

            successful_files += 1

            print()
            print(
                f"Documents : {file_documents}"
            )

            print(
                f"Chunks    : {file_chunks}"
            )

            print(
                f"Vectors   : {file_vectors}"
            )

            print(
                "Status    : SUCCESS"
            )

        except Exception as error:

            failed_files += 1

            error_info = {
                "file": str(file_path),
                "error": str(error),
            }

            errors.append(
                error_info
            )

            print()
            print(
                "Status    : FAILED"
            )

            print(
                f"Error     : {error}"
            )

            traceback.print_exc()

            continue

    # --------------------------------------------------------
    # SAVE VECTOR INDEX
    # --------------------------------------------------------

    print()
    print(
        "Saving vector index..."
    )

    vector_store.save()

    # --------------------------------------------------------
    # SAVE ERROR LOG
    # --------------------------------------------------------

    if errors:

        error_log_path = (
            VECTOR_STORE_DIR /
            "index_build_errors.json"
        )

        with open(
            error_log_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                errors,
                file,
                indent=2,
                ensure_ascii=False
            )

        print()
        print(
            f"Error log saved to:"
        )

        print(
            f"  {error_log_path}"
        )

    # --------------------------------------------------------
    # FINAL STATISTICS
    # --------------------------------------------------------

    final_faiss_count = (
        vector_store.count()
    )

    final_metadata_count = (
        vector_store.metadata_count()
    )

    print()
    print("=" * 70)
    print("INDEX BUILD COMPLETE")
    print("=" * 70)

    print(
        f"Total files       : {total_files}"
    )

    print(
        f"Successful files  : {successful_files}"
    )

    print(
        f"Failed files      : {failed_files}"
    )

    print(
        f"Documents created : {total_documents}"
    )

    print(
        f"Chunks created    : {total_chunks}"
    )

    print(
        f"New vectors       : {total_vectors}"
    )

    print(
        f"FAISS vectors     : {final_faiss_count}"
    )

    print(
        f"SQLite records    : {final_metadata_count}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # VALIDATE CONSISTENCY
    # --------------------------------------------------------

    if (
        final_faiss_count
        != final_metadata_count
    ):

        print()
        print(
            "[WARNING] FAISS and SQLite counts "
            "do not match."
        )

    else:

        print()
        print(
            "[PASS] FAISS and SQLite are synchronized."
        )

    # --------------------------------------------------------
    # CLOSE STORE
    # --------------------------------------------------------

    vector_store.close()

    print()
    print(
        "Vector store closed."
    )

    print()
    print(
        "Real document index is ready."
    )


# ============================================================
# SAVE CHUNKS
# ============================================================

def save_chunks(
    chunks,
    output_dir: Path,
):

    for chunk in chunks:

        chunk_id = str(
            chunk.metadata.get(
                "chunk_id",
                "unknown_chunk"
            )
        )

        safe_name = (
            chunk_id
            .replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
            .replace(" ", "_")
        )

        output_file = (
            output_dir /
            f"{safe_name}.json"
        )

        counter = 1

        while output_file.exists():

            output_file = (
                output_dir /
                f"{safe_name}_{counter}.json"
            )

            counter += 1

        data = {
            "text": chunk.text,
            "metadata": chunk.metadata,
        }

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2
            )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    build_index()