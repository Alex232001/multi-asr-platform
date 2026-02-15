terraform {
  required_providers {
    yandex = {
      source  = "yandex-cloud/yandex"
      version = "~> 0.92" 
    }
  }
  required_version = ">= 0.13"

}

provider "yandex" {
  zone = "ru-central1-a"
  folder_id = var.folder_id

  service_account_key_file = "sa-key.json"
}

# Service Account для контейнеров
resource "yandex_iam_service_account" "container" {
  name        = "container-sa"
  description = "Service account for serverless containers"
}

resource "yandex_resourcemanager_folder_iam_member" "container_viewer" {
  folder_id = var.folder_id
  role      = "viewer"
  member    = "serviceAccount:${yandex_iam_service_account.container.id}"
}

resource "yandex_resourcemanager_folder_iam_member" "container_puller" {
  folder_id = var.folder_id
  role      = "container-registry.images.puller"
  member    = "serviceAccount:${yandex_iam_service_account.container.id}"
}

# Определяем контейнеры в map
locals {
  containers = {
    whisper-service = {
      name    = "whisper-service"
      image   = "cr.yandex/${var.registry_id}/whisper-service:${var.images_tag}"
      memory  = 2048
      cores   = 2
    }
    vosk-service = {
      name    = "vosk-service"
      image   = "cr.yandex/${var.registry_id}/vosk-service:${var.images_tag}"
      memory  = 512
      cores   = 1
    }
    nemo-service = {
      name    = "nemo-service"
      image   = "cr.yandex/${var.registry_id}/nemo-service:${var.images_tag}"
      memory  = 2048
      cores   = 1
    }
  }
}

# Serverless Containers 
resource "yandex_serverless_container" "containers" {
  for_each = local.containers

  name               = each.value.name
  memory             = each.value.memory
  cores              = each.value.cores 
  core_fraction      = 100
  execution_timeout  = "300s"
  concurrency        = 1
  service_account_id = yandex_iam_service_account.container.id

  image {
    url = each.value.image
  }
}


# Output всех URL контейнеров
output "container_urls" {
  description = "URLs всех контейнеров"
  value = {
    for name, container in yandex_serverless_container.containers :
    name => container.url
  }
}

# Output всех ID контейнеров
output "container_ids" {
  description = "IDs всех контейнеров"
  value = {
    for name, container in yandex_serverless_container.containers :
    name => container.id
  }
}
