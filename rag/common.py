"""Settings and helpers shared by the ingestion and search scripts."""

import os

REGION = os.environ.get("AWS_REGION", "eu-north-1")
DATA_BUCKET = "rag-portfolio-anas-data-2026"
DATA_PREFIX = "raw/arxiv/"
VECTOR_BUCKET = "rag-portfolio-anas-vectors-2026"
INDEX_NAME = "arxiv-abstracts"

# Free embedding model that runs on your own computer. It turns text into
# 384 numbers, which must match the dimension of the vector index.
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIMENSION = 384

# The slice of arXiv we care about: artificial intelligence papers.
AI_CATEGORIES = ("cs.AI", "cs.CL", "cs.LG", "stat.ML")


def get_embedder():
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=EMBED_MODEL)


def get_vectors_client():
    import boto3

    return boto3.client("s3vectors", region_name=REGION)
