output "data_bucket_name" {
  value = google_storage_bucket.data.name
}

output "models_bucket_name" {
  value = google_storage_bucket.models.name
}