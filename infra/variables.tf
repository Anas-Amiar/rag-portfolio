variable "region" {
  description = "AWS region where everything lives."
  type        = string
  default     = "eu-north-1"
}

variable "data_bucket_name" {
  description = "General purpose S3 bucket holding the raw arXiv files."
  type        = string
  default     = "rag-portfolio-anas-data-2026"
}

variable "vector_bucket_name" {
  description = "S3 vector bucket holding the embeddings."
  type        = string
  default     = "rag-portfolio-anas-vectors-2026"
}

variable "embedding_dimension" {
  description = "Length of each embedding. Must match the embedding model (384 for bge-small)."
  type        = number
  default     = 384
}
