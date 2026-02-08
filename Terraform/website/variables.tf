# ТОЛЬКО несекретные переменные
variable "folder_id" {
  type        = string
  description = "Yandex Cloud Folder ID"
  sensitive   = true
}

# дЛя теста потом убрать обязательно
variable "registry_id" {
  type        = string
  description = "Yandex Container Registry ID"
  sensitive   = true
}

variable "s3_bucket_name" {
  type        = string
  description = "S3 bucket name for Docker registry"
  default     = "docker-registry-stellarclaw"
}

variable "domain_name" {
  type        = string
  description = "Domain name"
  default     = "stellarclaw.ru"
}

variable "yc_access_key" {
  type        = string
  description = "Yandex Cloud Access Key for S3"
  sensitive   = true
}

variable "yc_secret_key" {
  type        = string
  description = "Yandex Cloud Secret Key for S3"
  sensitive   = true
}
