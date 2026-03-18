resource "google_storage_bucket" "data" {
  project       = var.project_id
  name          = var.data_bucket_name
  location      = var.location
  force_destroy = var.data_bucket_force_destroy
  labels        = var.labels

  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  versioning {
    enabled = true
  }
}

resource "google_storage_bucket" "models" {
  project       = var.project_id
  name          = var.models_bucket_name
  location      = var.location
  force_destroy = var.models_bucket_force_destroy
  labels        = var.labels

  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  versioning {
    enabled = true
  }
}