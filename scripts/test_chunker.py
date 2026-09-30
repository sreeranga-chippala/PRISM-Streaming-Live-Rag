from src.retrieval.document_loader import DocumentLoader
from src.retrieval.chunker import TextChunker


loader = DocumentLoader()

documents = loader.load(
    "data/raw/documents/test.txt"
)


chunker = TextChunker(
    chunk_size=20,
    chunk_overlap=5
)

chunks = chunker.chunk_documents(documents)


for chunk in chunks:

    print("=" * 70)

    print("CHUNK:")
    print(chunk.text)

    print("\nMETADATA:")
    print(chunk.metadata)