"""Ingestion: read arXiv papers, keep the AI ones, embed them, store them.

Run from the project folder:
    python rag/ingest.py --limit 200      # small test first
    python rag/ingest.py --limit 5000     # the real slice

Running it twice is safe: a paper with the same id just overwrites itself.
"""

import argparse
import glob
import io
import os

import pandas as pd

from common import (
    AI_CATEGORIES,
    DATA_BUCKET,
    DATA_PREFIX,
    EMBED_DIMENSION,
    INDEX_NAME,
    REGION,
    VECTOR_BUCKET,
    get_embedder,
    get_vectors_client,
)

COLUMNS = ["id", "title", "abstract", "categories"]
EMBED_BATCH = 256   # sentences turned into numbers at once
PUT_BATCH = 100     # vectors sent to AWS per request (the limit is 500)


def clean(text):
    """Collapse the line breaks and double spaces found in raw abstracts."""
    return " ".join(str(text).split())


def load_papers(source, filename=None):
    """Read one parquet file, either from this computer or from S3."""
    if source == "local":
        folder = os.path.expanduser("~/arxiv-data/data")
        path = os.path.join(folder, filename) if filename else sorted(
            glob.glob(os.path.join(folder, "*.parquet"))
        )[0]
        print(f"Reading {path}")
        return pd.read_parquet(path, columns=COLUMNS)

    import boto3

    s3 = boto3.client("s3", region_name=REGION)
    if filename:
        key = DATA_PREFIX + filename
    else:
        listing = s3.list_objects_v2(Bucket=DATA_BUCKET, Prefix=DATA_PREFIX)
        key = sorted(obj["Key"] for obj in listing["Contents"])[0]
    print(f"Reading s3://{DATA_BUCKET}/{key}")
    body = s3.get_object(Bucket=DATA_BUCKET, Key=key)["Body"].read()
    return pd.read_parquet(io.BytesIO(body), columns=COLUMNS)


def select_ai_papers(df, limit):
    """Keep papers that belong to at least one AI category, up to `limit`."""
    is_ai = df["categories"].fillna("").str.split().apply(
        lambda cats: any(c in AI_CATEGORIES for c in cats)
    )
    papers = df[is_ai].drop_duplicates(subset="id").head(limit).copy()
    papers["title"] = papers["title"].map(clean)
    papers["abstract"] = papers["abstract"].map(clean)
    return papers


def batches(items, size):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def to_vector_records(papers, vectors):
    """Pair each paper with its numbers, in the format S3 Vectors expects."""
    records = []
    for (_, paper), vector in zip(papers.iterrows(), vectors):
        records.append(
            {
                "key": paper["id"],
                "data": {"float32": [float(x) for x in vector]},
                "metadata": {
                    "title": paper["title"],
                    "abstract": paper["abstract"],
                    "categories": paper["categories"],
                },
            }
        )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=200, help="how many papers")
    parser.add_argument("--source", choices=["local", "s3"], default="local")
    parser.add_argument("--file", default=None, help="parquet file name")
    args = parser.parse_args()

    df = load_papers(args.source, args.file)
    print(f"{len(df):,} papers in the file")
    papers = select_ai_papers(df, args.limit)
    print(f"{len(papers):,} AI papers selected")

    model = get_embedder()
    client = get_vectors_client()

    stored = 0
    for chunk in batches(papers, EMBED_BATCH):
        texts = (chunk["title"] + ". " + chunk["abstract"]).tolist()
        vectors = list(model.embed(texts))
        assert len(vectors[0]) == EMBED_DIMENSION, "model and index sizes differ"
        records = to_vector_records(chunk, vectors)
        for part in batches(records, PUT_BATCH):
            client.put_vectors(
                vectorBucketName=VECTOR_BUCKET, indexName=INDEX_NAME, vectors=part
            )
            stored += len(part)
        print(f"  stored {stored:,} / {len(papers):,}")

    print("Done.")


if __name__ == "__main__":
    main()
