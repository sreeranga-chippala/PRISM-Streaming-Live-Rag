from pathlib import Path
import json
import traceback

from src.retrieval.document_loader import DocumentLoader
from src.retrieval.chunker import TextChunker


# ============================================================
# CONFIGURATION
# ============================================================

RAW_DIR = Path(
    "data/raw/documents"
)

OUTPUT_DIR = Path(
    "data/processed/chunks"
)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


# ============================================================
# INGESTION
# ============================================================

def ingest_documents():

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Clean previously generated chunk files.
    #
    # This makes ingestion reproducible:
    #
    # Run 1:
    #     chunk_a.json
    #     chunk_b.json
    #
    # Run 2:
    #     chunk_a.json
    #     chunk_b.json
    #
    # instead of:
    #     chunk_a.json
    #     chunk_a_1.json
    #     chunk_b.json
    #     chunk_b_1.json
    #
    # Only JSON files inside the generated chunks directory
    # are removed. Raw documents are never touched.
    # --------------------------------------------------------

    clean_output_directory(
        OUTPUT_DIR
    )

    loader = DocumentLoader()

    chunker = TextChunker(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    files = [
        path
        for path in RAW_DIR.rglob("*")
        if path.is_file()
        and not path.name.startswith(".")
    ]

    if not files:

        print(
            f"No files found in: {RAW_DIR}"
        )

        return

    print("=" * 70)
    print("DOCUMENT INGESTION")
    print("=" * 70)

    print(
        f"Input directory : {RAW_DIR}"
    )

    print(
        f"Output directory: {OUTPUT_DIR}"
    )

    print(
        f"Files found     : {len(files)}"
    )

    print("=" * 70)

    total_files = 0
    successful_files = 0
    failed_files = 0
    total_documents = 0
    total_chunks = 0

    errors = []

    # ========================================================
    # PROCESS FILES ONE AT A TIME
    # ========================================================

    for file_path in files:

        total_files += 1

        print()
        print("-" * 70)
        print(f"Processing: {file_path}")
        print("-" * 70)

        file_chunks = 0
        file_documents = 0

        try:

            # ------------------------------------------------
            # Load incrementally
            # ------------------------------------------------

            documents = loader.iter_load(
                file_path
            )

            # ------------------------------------------------
            # Chunk incrementally
            # ------------------------------------------------

            for document in documents:

                file_documents += 1
                total_documents += 1

                for chunk in chunker.chunk_document(
                    document
                ):

                    save_chunk(
                        chunk,
                        OUTPUT_DIR
                    )

                    file_chunks += 1
                    total_chunks += 1

            successful_files += 1

            print(
                f"Documents: {file_documents}"
            )

            print(
                f"Chunks   : {file_chunks}"
            )

            print(
                "Status   : SUCCESS"
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

            print(
                "Status   : FAILED"
            )

            print(
                f"Error    : {error}"
            )

            # Important:
            # Continue with the next file.
            traceback.print_exc()

            continue

    # ========================================================
    # SAVE ERROR LOG
    # ========================================================

    error_log_path = (
        OUTPUT_DIR /
        "ingestion_errors.json"
    )

    if errors:

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
            f"Error log saved to: "
            f"{error_log_path}"
        )

    else:

        # ----------------------------------------------------
        # Remove an old error log if the current ingestion
        # completed without errors.
        # ----------------------------------------------------

        if error_log_path.exists():
            error_log_path.unlink()

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("INGESTION COMPLETE")
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

    print("=" * 70)


# ============================================================
# CLEAN OUTPUT DIRECTORY
# ============================================================

def clean_output_directory(
    output_dir: Path
):
    """
    Remove previously generated JSON files.

    The processed chunks directory contains generated
    ingestion artifacts, so clearing these files before a
    fresh ingestion makes the result reproducible.

    No raw documents are deleted.
    """

    removed_files = 0

    for path in output_dir.glob("*.json"):

        if path.is_file():

            path.unlink()

            removed_files += 1

    if removed_files > 0:

        print(
            f"Cleaned {removed_files} "
            f"previous generated JSON file(s)."
        )


# ============================================================
# SAVE CHUNK
# ============================================================

def save_chunk(
    chunk,
    output_dir: Path
):

    # --------------------------------------------------------
    # Create a stable filename
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Store chunk
    #
    # If the same chunk is processed again, the same filename
    # is used. There is intentionally NO _1, _2, _3 behavior.
    # --------------------------------------------------------

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

    ingest_documents()