output "data_bucket_name" {
  value = aws_s3_bucket.data.bucket
}

output "vector_bucket_name" {
  value = aws_s3vectors_vector_bucket.vectors.vector_bucket_name
}

output "vector_index_name" {
  value = aws_s3vectors_index.abstracts.index_name
}
