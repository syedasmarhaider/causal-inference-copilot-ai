terraform {
  required_providers {
    google-beta = {
      source = "hashicorp/google-beta"
    }
  }
}

resource "google_firebase_project" "firebase" {
  provider = google-beta
  project  = var.project_id
}

resource "google_firebase_database_instance" "database" {
  provider    = google-beta
  project     = google_firebase_project.firebase.project
  region      = var.database_region
  instance_id = var.database_instance_id
  type        = var.database_type
}
