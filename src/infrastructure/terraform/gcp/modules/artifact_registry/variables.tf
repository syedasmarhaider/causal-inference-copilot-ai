variable "project_id" {
  type = string
}

variable "location" {
  type = string
}

variable "repository_id" {
  type = string
}

variable "description" {
  type = string
}

variable "labels" {
  type    = map(string)
  default = {}
}