resource "google_service_account" "runtime" {
  project      = var.project_id
  account_id   = var.account_id
  display_name = var.display_name
}

resource "google_secret_manager_secret_iam_member" "secret_access" {
  for_each = var.secret_ids

  project   = var.project_id
  secret_id = each.value
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.runtime.email}"
}

resource "google_storage_bucket_iam_member" "bucket_access" {
  for_each = var.bucket_roles

  bucket = each.value.bucket
  role   = each.value.role
  member = "serviceAccount:${google_service_account.runtime.email}"
}

resource "google_project_iam_member" "project_roles" {
  for_each = var.project_roles

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.runtime.email}"
}

resource "google_service_account_iam_member" "service_account_user" {
  for_each = var.service_account_user_members

  service_account_id = google_service_account.runtime.name
  role               = "roles/iam.serviceAccountUser"
  member             = each.value
}
