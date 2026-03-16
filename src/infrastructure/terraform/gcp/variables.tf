variable "project_id" {
  description = "Existing GCP project ID."
  type        = string
}

variable "region" {
  description = "Primary region for Cloud Run."
  type        = string

  validation {
    condition     = length(trimspace(var.region)) > 0
    error_message = "region must not be empty."
  }
}

variable "environment" {
  description = "Deployment environment (dev, prod, feature)."
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "prod", "feature"], lower(trimspace(var.environment)))
    error_message = "environment must be one of: dev, prod, feature."
  }
}

variable "name_prefix" {
  description = "Prefix used when deriving resource names per environment."
  type        = string
  default     = "causal"

  validation {
    condition = (
      length(trimspace(var.name_prefix)) > 0
      && length(regexall("^[a-z][a-z0-9-]*$", var.name_prefix)) > 0
    )
    error_message = "name_prefix must start with a lowercase letter and contain only lowercase letters, digits, and '-'."
  }
}

variable "feature_slug" {
  description = "Short feature identifier used when environment=feature."
  type        = string
  default     = ""

  validation {
    condition = (
      lower(trimspace(var.environment)) != "feature"
      || (
        length(trimspace(var.feature_slug)) > 0
        && length(var.feature_slug) <= 20
        && length(regexall("^[a-z0-9-]+$", lower(var.feature_slug))) > 0
        && length(regexall("[a-z0-9]", lower(var.feature_slug))) > 0
      )
    )
    error_message = "feature_slug must be <= 20 chars and contain only lowercase letters, digits, and '-'."
  }
}

variable "labels" {
  description = "Common labels applied where supported."
  type        = map(string)
  default     = {}
}

variable "container_image" {
  description = "Full image reference for Cloud Run, ideally pinned by digest."
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
    condition = (
      contains(["DEFAULT_DATABASE", "USER_DATABASE"], var.firebase_database_type)
      && (lower(trimspace(var.environment)) != "feature" || var.firebase_database_type == "USER_DATABASE")
    )
    error_message = "firebase_database_type must be DEFAULT_DATABASE or USER_DATABASE; feature environments must use USER_DATABASE."
  }
}

variable "firebase_database_instance_id" {
  description = "Firebase RTDB instance ID. If null, a sensible default is derived."
  type        = string
  default     = null
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

variable "cloud_run_cpu" {
  description = "Cloud Run CPU limit."
  type        = string
  default     = "1"
}

variable "cloud_run_memory" {
  description = "Cloud Run memory limit."
  type        = string
  default     = "2Gi"
}

variable "cloud_run_concurrency" {
  description = "Cloud Run max concurrent requests per instance."
  type        = number
  default     = 10

  validation {
    condition     = var.cloud_run_concurrency >= 1 && var.cloud_run_concurrency <= 1000
    error_message = "cloud_run_concurrency must be between 1 and 1000."
  }
}

variable "cloud_run_timeout_seconds" {
  description = "Cloud Run request timeout in seconds."
  type        = number
  default     = 300

  validation {
    condition     = var.cloud_run_timeout_seconds >= 1 && var.cloud_run_timeout_seconds <= 3600
    error_message = "cloud_run_timeout_seconds must be between 1 and 3600."
  }
}

variable "cloud_run_min_instances" {
  description = "Cloud Run minimum instance count."
  type        = number
  default     = 0

  validation {
    condition     = var.cloud_run_min_instances >= 0
    error_message = "cloud_run_min_instances must be >= 0."
  }
}

variable "cloud_run_max_instances" {
  description = "Cloud Run maximum instance count."
  type        = number
  default     = 1

  validation {
    condition     = var.cloud_run_max_instances >= var.cloud_run_min_instances
    error_message = "cloud_run_max_instances must be >= cloud_run_min_instances."
  }
}

variable "cloud_run_container_port" {
  description = "Container port exposed by the app."
  type        = number
  default     = 8080

  validation {
    condition     = var.cloud_run_container_port >= 1 && var.cloud_run_container_port <= 65535
    error_message = "cloud_run_container_port must be between 1 and 65535."
  }
}

variable "cloud_run_ingress" {
  description = "Cloud Run ingress policy."
  type        = string
  default     = "INGRESS_TRAFFIC_ALL"

  validation {
    condition = contains([
      "INGRESS_TRAFFIC_ALL",
      "INGRESS_TRAFFIC_INTERNAL_ONLY",
      "INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER",
    ], var.cloud_run_ingress)
    error_message = "cloud_run_ingress must be one of INGRESS_TRAFFIC_ALL, INGRESS_TRAFFIC_INTERNAL_ONLY, or INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER."
  }
}

variable "cloud_run_allow_unauthenticated" {
  description = "Whether to allow public unauthenticated access to the Cloud Run service."
  type        = bool
  default     = false
}

variable "cloud_run_deletion_protection" {
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

variable "langfuse_public_key" {
  description = "Langfuse public key (non-secret)."
  type        = string
  default     = ""
}

variable "gcs_models_timeout_seconds" {
  description = "Model GCS read timeout."
  type        = number
  default     = 60

  validation {
    condition     = var.gcs_models_timeout_seconds > 0
    error_message = "gcs_models_timeout_seconds must be > 0."
  }
}

variable "gcs_models_upload_timeout_seconds" {
  description = "Model upload timeout."
  type        = number
  default     = 300

  validation {
    condition     = var.gcs_models_upload_timeout_seconds > 0
    error_message = "gcs_models_upload_timeout_seconds must be > 0."
  }
}

variable "gcs_models_upload_retry_timeout_seconds" {
  description = "Model upload retry timeout."
  type        = number
  default     = 900

  validation {
    condition     = var.gcs_models_upload_retry_timeout_seconds > 0
    error_message = "gcs_models_upload_retry_timeout_seconds must be > 0."
  }
}

variable "gcs_models_upload_chunk_size_bytes" {
  description = "Model upload chunk size."
  type        = number
  default     = 8388608

  validation {
    condition     = var.gcs_models_upload_chunk_size_bytes > 0 && floor(var.gcs_models_upload_chunk_size_bytes) % (256 * 1024) == 0
    error_message = "gcs_models_upload_chunk_size_bytes must be a positive multiple of 262144."
  }
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
