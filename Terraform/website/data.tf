
data "terraform_remote_state" "microservice" {
  backend = "s3"

  config = {
    endpoints = {
      s3 = "https://storage.yandexcloud.net"
    }
    bucket                      = "terraform-state-bucket-alex13"
    region                      = "ru-central1"
    key                         = "microservice/terraform.tfstate"
    skip_region_validation      = true
    skip_credentials_validation = true
    skip_requesting_account_id  = true
    skip_s3_checksum            = true

    access_key = var.yc_access_key
    secret_key = var.yc_secret_key
  }
}
