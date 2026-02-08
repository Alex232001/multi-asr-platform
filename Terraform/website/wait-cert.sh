#!/bin/bash
set +e 

CERT_NAME=$1
TIMEOUT=${2:-1200}

# Проверяем обязательный параметр
if [ -z "$CERT_NAME" ]; then
    echo "ERROR: Certificate name is required!" >&2
    echo "Usage: $0 <certificate_name> [timeout]" >&2
    exit 1
fi

END_TIME=$((SECONDS + TIMEOUT))

echo "Ожидание готовности сертификата: $CERT_NAME (timeout: ${TIMEOUT}с)" >&2

while [ $SECONDS -lt $END_TIME ]; do
    STATUS=$(yc certificate-manager certificate get "$CERT_NAME" --format json 2>/dev/null | jq -r '.status')
    
    if [ "$STATUS" = "ISSUED" ] || [ "$STATUS" = "VALID" ]; then
        echo "Сертификат готов!" >&2
        exit 0
    elif [ "$STATUS" = "VALIDATING" ]; then
        echo "Сертификат еще проверяется..." >&2
        sleep 10
    elif [ -z "$STATUS" ]; then
        echo "Certificate $CERT_NAME not found or already deleted" >&2
        exit 0 
    else
        echo "Неизвестный статус: $STATUS" >&2
        exit 1
    fi
done

echo "Timeout! Сертификат не готов за $TIMEOUT секунд" >&2
exit 1