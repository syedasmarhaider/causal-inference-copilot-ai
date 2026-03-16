variable "project_id" {
  type = string
}

variable "location" {
  type = string
}

variable "data_bucket_name" {
  type = string
}

variable "models_bucket_name" {
  type = string
}

variable "data_bucket_force_destroy" {
  type    = bool
  default = false
}

variable "models_bucket_force_destroy" {
  type    = bool
  default = false
}

variable "labels" {
  type    = map(string)
  default = {}
}