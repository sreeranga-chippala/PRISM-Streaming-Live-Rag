from src.retrieval.embeddings import EmbeddingModel
import numpy as np


def cosine_similarity(a, b):

    return np.dot(a, b)


def main():

    model = EmbeddingModel()

    text1 = (
        "How do I turn on Bluetooth?"
    )

    text2 = (
        "Open Settings and enable Bluetooth."
    )

    text3 = (
        "The weather is very cold today."
    )

    embedding1 = model.encode(text1)
    embedding2 = model.encode(text2)
    embedding3 = model.encode(text3)

    similarity_12 = cosine_similarity(
        embedding1,
        embedding2
    )

    similarity_13 = cosine_similarity(
        embedding1,
        embedding3
    )

    print()
    print(
        "Similarity between related sentences:",
        similarity_12
    )

    print(
        "Similarity between unrelated sentences:",
        similarity_13
    )


if __name__ == "__main__":
    main()