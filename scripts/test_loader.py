from src.retrieval.document_loader import DocumentLoader


loader = DocumentLoader()

files = [
    "data/raw/documents/test.txt",
    "data/raw/documents/test.json",
    "data/raw/documents/test.csv",
]


for file in files:

    print("\n" + "=" * 70)
    print(f"FILE: {file}")
    print("=" * 70)

    documents = loader.load(file)

    for i, document in enumerate(documents):

        print(f"\nDocument {i + 1}")

        print("TEXT:")
        print(document.text)

        print("\nMETADATA:")
        print(document.metadata)