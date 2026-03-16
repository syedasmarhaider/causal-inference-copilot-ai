locals {
  artifact_registry_location = coalesce(var.artifact_registry_location, var.region)
  buckets_location           = coalesce(var.buckets_location, var.region)

  firebase_database_instance_id = coalesce(
    var.firebase_database_instance_id,
    var.firebase_database_type == "DEFAULT_DATABASE" ? "${var.project_id}-default-rtdb" : "${var.service_name}-rtdb"
  )

  required_services = toset([
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "secretmanager.googleapis.com",
    "iam.googleapis.com",
    "serviceusage.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "storage.googleapis.com",
    "firebase.googleapis.com",
    "firebasedatabase.googleapis.com",
  ])
}

module "project_services" {
  source = "./modules/project_services"

  project_id = var.project_id
  services   = local.required_services
}

module "artifact_registry" {
  source = "./modules/artifact_registry"

  project_id    = var.project_id
  location      = local.artifact_registry_location
  repository_id = var.artifact_registry_repository_id
  description   = var.artifact_registry_description
  labels        = var.labels

  depends_on = [module.project_services]
}

module "storage_buckets" {
  source = "./modules/storage_buckets"

  project_id                  = var.project_id
  location                    = local.buckets_location
  data_bucket_name            = var.data_bucket_name
  models_bucket_name          = var.models_bucket_name
  data_bucket_force_destroy   = var.data_bucket_force_destroy
  models_bucket_force_destroy = var.models_bucket_force_destroy
  labels                      = var.labels

  depends_on = [module.project_services]
}

module "firebase_rtdb" {
  source = "./modules/firebase_rtdb"

  providers = {
    google-beta = google-beta
  }

  project_id           = var.project_id
  database_region      = var.firebase_database_region
  database_type        = var.firebase_database_type
  database_instance_id = local.firebase_database_instance_id

  depends_on = [module.project_services]
}

module "runtime_service_account" {
  source = "./modules/service_account"

  project_id   = var.project_id
  account_id   = var.runtime_service_account_id
  display_name = "${var.service_name} Cloud Run runtime"

  secret_ids = toset([
    for _, cfg in var.secret_env_vars : cfg.secret_id
  ])

  bucket_roles = {
    data = {
      bucket = module.storage_buckets.data_bucket_name
      role   = var.runtime_bucket_role
    }
    models = {
      bucket = module.storage_buckets.models_bucket_name
      role   = var.runtime_bucket_role
    }
  }

  project_roles = toset(var.runtime_extra_project_roles)

  depends_on = [
    module.project_services,
    module.storage_buckets
  ]
}

locals {
  plain_env_vars = merge(
    {
      LANGFUSE_BASE_URL                       = var.langfuse_base_url
      LANGFUSE_DEBUG                          = var.langfuse_debug
      GOOGLE_CLOUD_PROJECT_ID                 = var.project_id
      FIREBASE_DATABASE_URL                   = module.firebase_rtdb.database_url
      GCS_MODELS_BUCKET_NAME                  = module.storage_buckets.models_bucket_name
      GCS_DATA_BUCKET_NAME                    = module.storage_buckets.data_bucket_name
      GCS_MODELS_TIMEOUT_SECONDS              = tostring(var.gcs_models_timeout_seconds)
      GCS_MODELS_UPLOAD_TIMEOUT_SECONDS       = tostring(var.gcs_models_upload_timeout_seconds)
      GCS_MODELS_UPLOAD_RETRY_TIMEOUT_SECONDS = tostring(var.gcs_models_upload_retry_timeout_seconds)
      GCS_MODELS_UPLOAD_CHUNK_SIZE_BYTES      = tostring(var.gcs_models_upload_chunk_size_bytes)
    },
    var.extra_plain_env_vars
  )
}

module "cloud_run_service" {
  source = "./modules/cloud_run_service"

  project_id               = var.project_id
  region                   = var.region
  service_name             = var.service_name
  container_image          = var.container_image
  service_account_email    = module.runtime_service_account.email
  env_vars                 = local.plain_env_vars
  secret_env_vars          = var.secret_env_vars
  cpu                      = var.cpu
  memory                   = var.memory
  concurrency              = var.concurrency
  timeout_seconds          = var.timeout_seconds
  min_instances            = var.min_instances
  max_instances            = var.max_instances
  container_port           = var.container_port
  ingress                  = var.ingress
  allow_unauthenticated    = var.allow_unauthenticated
  deletion_protection      = var.deletion_protection
  labels                   = var.labels

  depends_on = [
    module.project_services,
    module.runtime_service_account,
    module.firebase_rtdb
  ]
}