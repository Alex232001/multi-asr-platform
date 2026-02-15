
variable "folder_id" {
  type        = string
  description = "Yandex Cloud Folder ID"
  sensitive   = true
}

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

variable "container_memory" {
  type        = number
  description = "Memory for serverless container (MB)"
  default     = 2048
}


#variable "vosk_images" {
#  type        = string
#  description = "Images for serverless"
#  default     = "whisper-service"
#}

variable "images_tag" {
  type        = string
  description = "Images tag for serverless"
  default     = "v1.1.0"
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


//variable "yc_service_account_key_json" {
//  type        = string
 // description = "JSON ключ сервисного аккаунта"
//  sensitive   = true
//}