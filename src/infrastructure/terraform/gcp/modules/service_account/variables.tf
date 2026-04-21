variable "project_id" {
  type = string
}

variable "account_id" {
  type = string
}

variable "display_name" {
  type = string
}

variable "secret_ids" {
  type    = set(string)
  default = []
}

variable "bucket_roles" {
  type = map(object({
    bucket = string
    role   = string
  }))
  default = {}
}

variable "project_roles" {
  type    = set(string)
  default = []
}

variable "service_account_user_members" {
  type    = set(string)
  default = []
}
