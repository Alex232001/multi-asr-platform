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


locals {
  #container_ids = data.terraform_remote_state.microservice.outputs.container_ids
  #container_urls = data.terraform_remote_state.microservice.outputs.container_urls

  container_ids = try(
    data.terraform_remote_state.microservice.outputs.container_ids,
    {
      "whisper-service" = "d9q96d73r1ujs86t2vbd"
      "vosk-service"    = "e8q85c62q0tir75u1uac"
      "nemo-service"    = "f7q74b51p9shq64t0tzb"
    }
  )

  container_urls = try(
    data.terraform_remote_state.microservice.outputs.container_urls,
    {
      "whisper-service" = "https://fake.url"
      "vosk-service"    = "https://fake.url"
      "nemo-service"    = "https://fake.url"
    }
  )


}



resource "yandex_storage_bucket" "website" {
  bucket = "service.stellarclaw.ru"

  anonymous_access_flags {
    read        = true
    list        = false
    config_read = false
  }
  website {
    index_document = "index.html"
    error_document = "error.html"
  }
}

resource "yandex_storage_object" "index" {
  bucket = yandex_storage_bucket.website.bucket
  key    = "index.html"
  source = "index.html"
  acl    = "public-read"
  content_type = "text/html"

  source_hash = filemd5("index.html")
}





resource "yandex_storage_object" "js_files" {
  for_each = fileset("${path.module}/js/", "**")

  bucket = yandex_storage_bucket.website.bucket
  key    = "js/${each.value}"
  source = "${path.module}/js/${each.value}"
  acl    = "public-read"
  content_type = "application/javascript"

  source_hash = filemd5("${path.module}/js/${each.value}")
}



resource "local_file" "api_config" {
  filename = "${path.module}/js/api-config.js"
  content  = <<-EOT
const API_ENDPOINTS = {
${join(",\n", [
  for name, id in local.container_ids :
  "    '${replace(name, "-service", "")}': 'https://service.stellarclaw.ru/api/transcribe/${replace(name, "-service", "")}'"
])}
};

// Экспорт
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { API_ENDPOINTS };
}
EOT
}



resource "yandex_storage_object" "api_config_js" {
  bucket = yandex_storage_bucket.website.bucket
  key    = "js/api-config.js"
  source = "${path.module}/js/api-config.js"
  acl    = "public-read"
  content_type = "application/javascript"

  # Перезагружаем, если содержимое изменилось
  #source_hash = filemd5("${path.module}/js/api-config.js")

  #  файл создаётся до загрузки
  depends_on = [local_file.api_config]
}




resource "yandex_storage_object" "css_files" {
  for_each = fileset("${path.module}/styles/", "**")

  bucket = yandex_storage_bucket.website.bucket
  key    = "styles/${each.value}"
  source = "${path.module}/styles/${each.value}"
  acl    = "public-read"
  content_type = "text/css"

  source_hash = filemd5("${path.module}/styles/${each.value}")
}



resource "yandex_iam_service_account" "api" {
  name        = "api-gateway-sa"
  description = "Service account for API Gateway to invoke containers"
}


resource "yandex_resourcemanager_folder_iam_member" "container_invoker" {
  folder_id = var.folder_id
  role      = "serverless.containers.invoker"
  member    = "serviceAccount:${yandex_iam_service_account.api.id}"
}


resource "yandex_iam_service_account" "website_sa" {
  name = "website-sa"
}

resource "yandex_resourcemanager_folder_iam_member" "storage_viewer" {
  folder_id = var.folder_id
  role      = "storage.viewer"
  member    = "serviceAccount:${yandex_iam_service_account.website_sa.id}"
}





resource "yandex_dns_zone" "stellarclaw_zone" {
  name        = "stellarclaw-zone"
  zone        = "stellarclaw.ru."
  public      = true
}

resource "yandex_cm_certificate" "website_cert" {
  name        = "website-stellarclaw-cert"
  description = "SSL certificate for service.stellarclaw.ru"
  domains     = ["service.stellarclaw.ru"]

  managed {
    challenge_type = "DNS_CNAME"
  }

  lifecycle {
    create_before_destroy = true
  }
}

resource "yandex_dns_recordset" "acme_challenge" {
  zone_id = yandex_dns_zone.stellarclaw_zone.id
  name    = yandex_cm_certificate.website_cert.challenges[0].dns_name
  type    = "CNAME"
  ttl     = 60
  data    = [yandex_cm_certificate.website_cert.challenges[0].dns_value]


  depends_on = [yandex_cm_certificate.website_cert]
}

#data "external" "wait_for_certificate" {
#  program = ["bash", "${path.module}/wait-for-certificate.sh"]
#  query = {
#    certificate_name = yandex_cm_certificate.website_cert.name
#    timeout          = "1200" # 20 минут
    #service_account_key = filebase64("sa-key.json") #
#  }
#  depends_on = [yandex_dns_recordset.acme_challenge]
#}

resource "null_resource" "wait_for_certificate" {
  triggers = {
    # Триггеры для перезапуска при изменении
    cert_name = yandex_cm_certificate.website_cert.name
    dns_name  = yandex_dns_recordset.acme_challenge.name
    timestamp = timestamp()  
  }
  
  provisioner "local-exec" {
    command = <<-EOT
      chmod +x ${path.module}/wait-cert.sh
      ${path.module}/wait-cert.sh \
        "${yandex_cm_certificate.website_cert.name}" \
        "1200"
    EOT
    
    when = create
  }
  
  provisioner "local-exec" {
    command = "echo 'Skipping certificate destroy'"
    when = destroy
    
    on_failure = continue
  }
  
  depends_on = [yandex_dns_recordset.acme_challenge]
}


resource "yandex_api_gateway" "website_gateway" {
  name        = "website-gateway"
  description = "Website API Gateway"
  folder_id   = var.folder_id

  depends_on = [null_resource.wait_for_certificate]
  #depends_on = [data.external.wait_for_certificate]

  custom_domains {
    fqdn           = "service.stellarclaw.ru"
    certificate_id = yandex_cm_certificate.website_cert.id
  }

  spec = <<-EOT
    openapi: "3.0.0"
    info:
      version: 1.0.0
      title: Website Gateway
    paths:
      /:
        get:
          x-yc-apigateway-integration:
            type: object_storage
            bucket: "${yandex_storage_bucket.website.bucket}"
            object: "index.html"
            service_account_id: "${yandex_iam_service_account.website_sa.id}"
      /{path+}:
        get:
          parameters:
            - name: path
              in: path
              required: true
              schema:
                type: string
          x-yc-apigateway-integration:
            type: object_storage
            bucket: "${yandex_storage_bucket.website.bucket}"
            object: "{path}"
            service_account_id: "${yandex_iam_service_account.website_sa.id}"
      /api/transcribe/vosk:
        post:
          x-yc-apigateway-integration:
            type: http
            url: https://${local.container_ids["vosk-service"]}.containers.yandexcloud.net/transcribe
            service_account_id: "${yandex_iam_service_account.api.id}"
          responses:
            "200":
              description: OK
      /api/transcribe/whisper:
        post:
          x-yc-apigateway-integration:
            type: http
            url: https://${local.container_ids["whisper-service"]}.containers.yandexcloud.net/transcribe
            service_account_id: "${yandex_iam_service_account.api.id}"
          responses:
            "200":
              description: OK
      /api/transcribe/nemo:
        post:
          x-yc-apigateway-integration:
            type: http
            url: https://${local.container_ids["nemo-service"]}.containers.yandexcloud.net/transcribe
            service_account_id: "${yandex_iam_service_account.api.id}"
          responses:
            "200":
              description: OK
  EOT
}

resource "yandex_dns_recordset" "website_alias" {
  zone_id = yandex_dns_zone.stellarclaw_zone.id
  name    = "service.stellarclaw.ru."
  type    = "CNAME"  
  ttl     = 60
  data    = ["${yandex_api_gateway.website_gateway.domain}."]
  #data    = [yandex_api_gateway.website_gateway.domain]

  depends_on = [yandex_api_gateway.website_gateway]
}

output "api_gateway_domain" {
  value = yandex_api_gateway.website_gateway.domain
}

output "website_url" {
  value = "https://${yandex_storage_bucket.website.bucket}.website.yandexcloud.net"
}


#locals {
#  container_urls = data.terraform_remote_state.microservice.outputs.container_urls
#
#  service_urls = {
#    for name, url in local.container_urls :
#    replace(name, "-service", "") => url
#  }
#}


