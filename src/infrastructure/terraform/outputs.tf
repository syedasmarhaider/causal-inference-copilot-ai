output "cloud_run_url" {
  value = module.cloud_run_service.url
}

output "cloud_run_service_name" {
  value = module.cloud_run_service.name
}

output "artifact_registry_repository_url" {
  value = module.artifact_registry.repository_url
}

output "data_bucket_name" {
  value = module.storage_buckets.data_bucket_name
}

output "models_bucket_name" {
  value = module.storage_buckets.models_bucket_name
}

output "firebase_database_instance_id" {
  value = module.firebase_rtdb.instance_id
}

output "firebase_database_url" {
  value = module.firebase_rtdb.database_url
}

output "runtime_service_account_email" {
  value = module.runtime_service_account.email
}