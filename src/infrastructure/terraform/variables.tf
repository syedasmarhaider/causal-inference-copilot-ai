variable "project_id" {
  description = "Existing GCP project ID."
  type        = string
}

variable "region" {
  description = "Primary region for Cloud Run."
  type        = string
}

variable "labels" {
  description = "Common labels applied where supported."
  type        = map(string)
  default     = {}
}

variable "service_name" {
  description = "Cloud Run service name."
  type        = string
}

variable "container_image" {
  description = "Full image reference for Cloud Run, ideally pinned by digest."
  type        = string
}

variable "artifact_registry_location" {
  description = "Artifact Registry location. Defaults to var.region when null."
  type        = string
  default     = null
}

variable "artifact_registry_repository_id" {
  description = "Artifact Registry Docker repository ID."
  type        = string
  default     = "backend-images"
}

variable "artifact_registry_description" {
  description = "Artifact Registry repository description."
  type        = string
  default     = "Docker repository for backend images"
}

variable "data_bucket_name" {
  description = "GCS bucket name for application data."
  type        = string
}

variable "models_bucket_name" {
  description = "GCS bucket name for model artifacts."
  type        = string
}

variable "buckets_location" {
  description = "Bucket location. Defaults to var.region when null."
  type        = string
  default     = null
}

variable "data_bucket_force_destroy" {
  description = "Allow destroy of non-empty data bucket."
  type        = bool
  default     = false
}

variable "models_bucket_force_destroy" {
  description = "Allow destroy of non-empty models bucket."
  type        = bool
  default     = false
}

variable "firebase_database_region" {
  description = "Firebase Realtime Database region."
  type        = string
}

variable "firebase_database_type" {
  description = "Firebase RTDB type: DEFAULT_DATABASE or USER_DATABASE."
  type        = string
  default     = "DEFAULT_DATABASE"

  validation {
    condition     = contains(["DEFAULT_DATABASE", "USER_DATABASE"], var.firebase_database_type)
    error_message = "firebase_database_type must be DEFAULT_DATABASE or USER_DATABASE."
  }
}

variable "firebase_database_instance_id" {
  description = "Firebase RTDB instance ID. If null, a sensible default is derived."
  type        = string
  default     = null
}

variable "runtime_service_account_id" {
  description = "Account ID (not email) for the Cloud Run runtime service account."
  type        = string
  default     = "aitia-backend"
}

variable "runtime_extra_project_roles" {
  description = "Extra project-level roles for the runtime service account."
  type        = list(string)
  default     = []
}

variable "runtime_bucket_role" {
  description = "Bucket-level IAM role granted to the runtime service account on both buckets."
  type        = string
  default     = "roles/storage.admin"
}

variable "cpu" {
  description = "Cloud Run CPU limit."
  type        = string
  default     = "1"
}

variable "memory" {
  description = "Cloud Run memory limit."
  type        = string
  default     = "2Gi"
}

variable "concurrency" {
  description = "Cloud Run max concurrent requests per instance."
  type        = number
  default     = 10
}

variable "timeout_seconds" {
  description = "Cloud Run request timeout in seconds."
  type        = number
  default     = 300
}

variable "min_instances" {
  description = "Cloud Run minimum instance count."
  type        = number
  default     = 0
}

variable "max_instances" {
  description = "Cloud Run maximum instance count."
  type        = number
  default     = 1
}

variable "container_port" {
  description = "Container port exposed by the app."
  type        = number
  default     = 8080
}

variable "ingress" {
  description = "Cloud Run ingress policy."
  type        = string
  default     = "INGRESS_TRAFFIC_ALL"
}

variable "allow_unauthenticated" {
  description = "Whether to allow public unauthenticated access to the Cloud Run service."
  type        = bool
  default     = false
}

variable "deletion_protection" {
  description = "Cloud Run deletion protection."
  type        = bool
  default     = false
}

variable "langfuse_base_url" {
  description = "Langfuse base URL."
  type        = string
  default     = "https://cloud.langfuse.com"
}

variable "langfuse_debug" {
  description = "Langfuse debug flag as a string."
  type        = string
  default     = "False"
}

variable "gcs_models_timeout_seconds" {
  description = "Model GCS read timeout."
  type        = number
  default     = 60
}

variable "gcs_models_upload_timeout_seconds" {
  description = "Model upload timeout."
  type        = number
  default     = 300
}

variable "gcs_models_upload_retry_timeout_seconds" {
  description = "Model upload retry timeout."
  type        = number
  default     = 900
}

variable "gcs_models_upload_chunk_size_bytes" {
  description = "Model upload chunk size."
  type        = number
  default     = 8388608
}

variable "extra_plain_env_vars" {
  description = "Additional non-sensitive env vars."
  type        = map(string)
  default     = {}
}

variable "secret_env_vars" {
  description = "Map of Cloud Run env var name to existing Secret Manager secret reference."
  type = map(object({
    secret_id = string
    version   = optional(string, "latest")
  }))
  default = {}
}