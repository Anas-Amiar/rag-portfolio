"""Search: ask a question, get the closest papers.

    python rag/search.py "How do I reduce hallucinations in language models?"
"""

import argparse

from common import INDEX_NAME, VECTOR_BUCKET, get_embedder, get_vectors_client


def find_papers(question, top_k=5, model=None, client=None):
    """Return the `top_k` papers closest in meaning to the question."""
    model = model or get_embedder()
    client = client or get_vectors_client()
    vector = [float(x) for x in list(model.query_embed([question]))[0]]
    result = client.query_vectors(
        vectorBucketName=VECTOR_BUCKET,
        indexName=INDEX_NAME,
        queryVector={"float32": vector},
        topK=top_k,
        returnMetadata=True,
        returnDistance=True,
    )
    return [
        {
            "id": hit["key"],
            "distance": hit["distance"],
            "title": hit["metadata"]["title"],
            "abstract": hit["metadata"]["abstract"],
            "url": f"https://arxiv.org/abs/{hit['key']}",
        }
        for hit in result["vectors"]
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    for rank, paper in enumerate(find_papers(args.question, args.top_k), start=1):
        print(f"{rank}. {paper['title']}")
        print(f"   distance {paper['distance']:.3f}  {paper['url']}")
        print(f"   {paper['abstract'][:200]}...\n")


if __name__ == "__main__":
    main()
