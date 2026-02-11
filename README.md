#  DevOps Infrastructure: Multi-Service Speech Recognition Platform

##  Project Overview
Инфраструктура для платформы распознавания речи на микросервисной архитектуре, развернутая в Yandex Cloud. Включает три AI-сервиса (Whisper, Vosk, Nemo) с автоматизированными CI/CD пайплайнами на Terraform и GitHub Actions.

## Architecture

```ascii
┌────────────────────────────────────────────────────────────────────────┐
│                          GitHub Repository                             │
│                    ┌─────────────────────────────┐                     │
│                    │    Infrastructure as Code    │                    │
│                    │   • Terraform configurations │                    │
│                    │   • Dockerfiles             │                     │
│                    │   • CI/CD workflows         │                     │
│                    └──────────────┬──────────────┘                     │
│                                   │                                    │
│                                   ▼                                    │
│                    ┌─────────────────────────────┐                     │
│                    │    GitHub Actions CI/CD      │                    │
│                    │   ┌─────────────────────┐   │                     │
│                    │   │  Matrix Build Job   │   │                     │
│                    │   │  • whisper-service  │   │                     │
│                    │   │  • vosk-service     │   │                     │
│                    │   │  • nemo-service     │   │                     │
│                    │   └─────────────────────┘   │                     │
│                    │   ┌─────────────────────┐   │                     │
│                    │   │ Terraform Deploy    │   │                     │
│                    │   │ • Serverless        │   │                     │
│                    │   │   Containers        │   │                     │
│                    │   │ • YC Resources      │   │                     │
│                    │   └─────────────────────┘   │                     │
│                    └──────────────┬──────────────┘                     │
│                                   │                                    │
└───────────────────────────────────┼────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Yandex Cloud (YC)                               │
│                    ┌─────────────────────────────┐                      │
│                    │   Container Registry (YCR)   │                     │
│                    │   • whisper-service:latest  │                      │
│                    │   • vosk-service:latest     │                      │
│                    │   • nemo-service:latest     │                      │
│                    └──────────────┬──────────────┘                      │
│                                   │                                     │
│                                   ▼                                     │
│                    ┌─────────────────────────────┐                      │
│                    │   Serverless Containers      │                     │
│                    │   ┌─────────────────────┐   │                      │
│                    │   │  whisper-service    │   │                      │
│                    │   │  • 2 vCPU, 2GB RAM  │   │                      │
│                    │   │  • Port: 8003       │   │                      │
│                    │   └─────────────────────┘   │                      │
│                    │   ┌─────────────────────┐   │                      │
│                    │   │  nemo-service       │   │                      │
│                    │   │  • 1 vCPU, 2GB RAM  │   │                      │
│                    │   │  • ONNX inference   │   │                      │
│                    │   └─────────────────────┘   │                      │
│                    │   ┌─────────────────────┐   │                      │
│                    │   │  vosk-service       │   │                      │
│                    │   │  • 1 vCPU, 512MB    │   │                      │
│                    │   │  • Vosk model       │   │                      │
│                    │   └─────────────────────┘   │                      │
│                    └─────────────────────────────┘                      │
└─────────────────────────────────────────────────────────────────────────┘
```
## Технологический стек

- **Инфраструктура**: Terraform с Yandex Cloud провайдером 
- **Контейнеризация**: Docker с многоступенчатыми сборками
- **Serverless платформа**: Yandex Cloud Serverless Containers
- **Управление секретами**: GitHub Secrets с JSON key аутентификацией
- **CI/CD**: GitHub Actions
- **Реестр образов**: Yandex Container Registry
- **Стратегия деплоя**: Blue-green через Terraform auto-approve
- **Управление конфигурацией**: Environment переменные + Terraform variables
- **Хранилище**: Object Storage для AI моделей (nemo, whisper)
- **Сеть**: Автоматический DNS, SSL endpoints, Api gateway
- **Сертификаты**: Yandex Certificate Manager (автоматически выпускает и обновляет SSL‑сертификаты)
- 

## Service Accounts

| Service Account | Где создается | Роли | Используется для |
|----------------|---------------|------|------------------|
| `container-sa` | microservice | `viewer`, `container-registry.images.puller` | Serverless Containers |
| `api-gateway-sa` | website | `serverless.containers.invoker` | API Gateway (вызов контейнеров) |
| `website-sa` | website | `storage.viewer` | API Gateway (чтение из Object Storage) |
| `github-actions` | | `container-registry.images.pusher` | Push в репозитории YC


## CI/CD Pipeline

### Workflow 1: Terraform Plan
- **Триггер**: pull_request
```yaml
on:
  pull_request:
    branches: [main]
  workflow_dispatch:
```
### Workflow 2: Build images
- **Триггер**: pull_request
- **Доп пояснения**: Отправка в Registry если выполняется merges если нет то просто build 
```yaml
name: Build-images
on:
  pull_request:
    branches: [main]
    types: [opened, synchronize, reopened, closed] 
  workflow_dispatch:
```

### Workflow 3: Deploy Terraform 
- **Триггер**: Ручной запуск 
- **Доп пояснения**: Deploy при подтвержении (ручном запуске) workflow_dispatch
```yaml
name: Deploy Terraform
on:
  workflow_dispatch:
```

### Workflow 4: Terraform Destroy
- **Триггер**: Ручное удаление 
```yaml
name: Terraform Destroy
on:
  workflow_dispatch:
```




