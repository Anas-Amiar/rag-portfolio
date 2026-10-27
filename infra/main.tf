# The data bucket already exists (we created it by hand and uploaded the papers).
# This import block tells Terraform to adopt it instead of creating a new one.
import {
  to = aws_s3_bucket.data
  id = var.data_bucket_name
}

resource "aws_s3_bucket" "data" {
  bucket = var.data_bucket_name

  # Lets "terraform destroy" delete the bucket even when it still holds files.
  # The data is public and can be downloaded again, so this is what we want here.
  force_destroy = true
}

resource "aws_s3_bucket_public_access_block" "data" {
  bucket = aws_s3_bucket.data.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Where the embeddings live.
resource "aws_s3vectors_vector_bucket" "vectors" {
  vector_bucket_name = var.vector_bucket_name
  force_destroy      = true
}

# The searchable shelf inside the vector bucket.
resource "aws_s3vectors_index" "abstracts" {
  index_name         = "arxiv-abstracts"
  vector_bucket_name = aws_s3vectors_vector_bucket.vectors.vector_bucket_name

  data_type       = "float32"
  dimension       = var.embedding_dimension
  distance_metric = "cosine"
}
