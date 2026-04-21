locals {
  environment_normalized = lower(trimspace(var.environment))
  feature_slug_sanitized = trim(replace(lower(var.feature_slug), "/[^a-z0-9-]/", "-"), "-")
  env_suffix = (
    local.environment_normalized == "feature"
    ? "feat-${local.feature_slug_sanitized}"
    : local.environment_normalized
  )
  resource_prefix = "${var.name_prefix}-${local.env_suffix}"
  project_env_prefix = (
    endswith(var.project_id, "-${local.env_suffix}")
    ? var.project_id
    : "${var.project_id}-${local.env_suffix}"
  )
  vertex_ai_project_id       = coalesce(var.vertex_ai_project_id, var.project_id)
  vertex_ai_location         = coalesce(var.vertex_ai_location, var.region)
  service_name               = "${local.resource_prefix}-backend"
  data_bucket_name           = "${local.project_env_prefix}-data"
  models_bucket_name         = "${local.project_env_prefix}-models"
  runtime_sa_id_base         = trim(replace(local.resource_prefix, "/[^a-z0-9-]/", "-"), "-")
  runtime_service_account_id = "${substr(local.runtime_sa_id_base, 0, 27)}-sa"
  vertex_runtime_role_id     = trim(replace(var.vertex_runtime_custom_role_id, "/[^A-Za-z0-9_]/", ""), "_")
  vertex_runtime_role_name   = "projects/${var.project_id}/roles/${local.vertex_runtime_role_id}"
  effective_labels           = merge(var.labels, { env = local.env_suffix })
  runtime_project_roles      = toset(concat(var.runtime_extra_project_roles, [local.vertex_runtime_role_name]))

  buckets_location = coalesce(var.buckets_location, var.region)

  firebase_database_instance_id = coalesce(
    var.firebase_database_instance_id,
    var.firebase_database_type == "DEFAULT_DATABASE" ? "${var.project_id}-default-rtdb" : "${local.service_name}-rtdb"
  )

  required_services = toset([
    "run.googleapis.com",
    "aiplatform.googleapis.com",
    "secretmanager.googleapis.com",
    "iam.googleapis.com",
    "serviceusage.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "storage.googleapis.com",
    "firebase.googleapis.com",
    "firebasedatabase.googleapis.com",
  ])
}

resource "google_project_iam_custom_role" "vertex_runtime" {
  project     = var.project_id
  role_id     = local.vertex_runtime_role_id
  title       = var.vertex_runtime_custom_role_title
  description = "Least-privilege Vertex AI runtime role for the Cloud Run backend."
  permissions = sort(tolist(var.vertex_runtime_custom_role_permissions))
  stage       = "GA"
}

module "project_services" {
  source = "./modules/project_services"

  project_id = var.project_id
  services   = local.required_services
}

module "storage_buckets" {
  source = "./modules/storage_buckets"

  project_id                  = var.project_id
  location                    = local.buckets_location
  data_bucket_name            = local.data_bucket_name
  models_bucket_name          = local.models_bucket_name
  data_bucket_force_destroy   = var.data_bucket_force_destroy
  models_bucket_force_destroy = var.models_bucket_force_destroy
  labels                      = local.effective_labels

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
  account_id   = local.runtime_service_account_id
  display_name = "${local.service_name} Cloud Run runtime"

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

  project_roles                = local.runtime_project_roles
  service_account_user_members = var.runtime_service_account_user_members

  depends_on = [
    module.project_services,
    module.storage_buckets,
    google_project_iam_custom_role.vertex_runtime,
  ]
}

locals {
  plain_env_vars = merge(
    {
      LITELLM_PROVIDER                        = "vertex_ai"
      LANGFUSE_BASE_URL                       = var.langfuse_base_url
      LANGFUSE_DEBUG                          = var.langfuse_debug
      LANGFUSE_PUBLIC_KEY                     = var.langfuse_public_key
      GOOGLE_CLOUD_PROJECT_ID                 = var.project_id
      VERTEXAI_PROJECT                        = local.vertex_ai_project_id
      VERTEXAI_LOCATION                       = local.vertex_ai_location
      FIREBASE_DATABASE_URL                   = module.firebase_rtdb.database_url
      GCS_MODELS_BUCKET_NAME                  = module.storage_buckets.models_bucket_name
      GCS_DATA_BUCKET_NAME                    = module.storage_buckets.data_bucket_name
      GCS_MODELS_TIMEOUT_SECONDS              = tostring(var.gcs_models_timeout_seconds)
      GCS_MODELS_UPLOAD_TIMEOUT_SECONDS       = tostring(var.gcs_models_upload_timeout_seconds)
      GCS_MODELS_UPLOAD_RETRY_TIMEOUT_SECONDS = tostring(var.gcs_models_upload_retry_timeout_seconds)
      GCS_MODELS_UPLOAD_CHUNK_SIZE_BYTES      = tostring(var.gcs_models_upload_chunk_size_bytes)
    },
    var.extra_plain_env_vars,
    {
      LITELLM_PROVIDER  = "vertex_ai"
      VERTEXAI_PROJECT  = local.vertex_ai_project_id
      VERTEXAI_LOCATION = local.vertex_ai_location
    }
  )
}

module "cloud_run_service" {
  source = "./modules/cloud_run_service"

  project_id            = var.project_id
  region                = var.region
  service_name          = local.service_name
  container_image       = var.container_image
  service_account_email = module.runtime_service_account.email
  env_vars              = local.plain_env_vars
  secret_env_vars       = var.secret_env_vars
  cpu                   = var.cloud_run_cpu
  memory                = var.cloud_run_memory
  concurrency           = var.cloud_run_concurrency
  timeout_seconds       = var.cloud_run_timeout_seconds
  min_instances         = var.cloud_run_min_instances
  max_instances         = var.cloud_run_max_instances
  container_port        = var.cloud_run_container_port
  ingress               = var.cloud_run_ingress
  allow_unauthenticated = var.cloud_run_allow_unauthenticated
  deletion_protection   = var.cloud_run_deletion_protection
  labels                = local.effective_labels

  depends_on = [
    module.project_services,
    module.runtime_service_account,
    module.firebase_rtdb
  ]
}
