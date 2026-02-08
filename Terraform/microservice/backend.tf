

terraform {
  backend "s3" {
    endpoints = {
      s3 = "https://storage.yandexcloud.net"
    }
    bucket                      = "terraform-state-bucket-alex13"
    region                      = "ru-central1"
    key                         = "microservice/terraform.tfstate"
    #access_key                  = var.yc_access_key
    #secret_key                  = var.yc_secret_key
    skip_region_validation      = true
    skip_credentials_validation = true
    skip_requesting_account_id  = true # Необходимая опция Terraform для версии 1.6.1 и старше.
    skip_s3_checksum            = true # Необходимая опция при описании бэкенда для Terraform версии 1.6.3 и старше.

  }
}






