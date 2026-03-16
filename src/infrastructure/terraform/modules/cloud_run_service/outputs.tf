output "name" {
  value = google_cloud_run_v2_service.service.name
}

output "url" {
  value = google_cloud_run_v2_service.service.uri
}