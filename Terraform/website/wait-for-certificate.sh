#!/bin/bash
set +e 
# Чтение входных параметров
eval "$(jq -r '@sh "CERT_NAME=\(.certificate_name) TIMEOUT=\(.timeout)"')"

# Установка timeout
END_TIME=$((SECONDS + TIMEOUT))

echo "Ожидание готовности сертификата: $CERT_NAME (timeout: ${TIMEOUT}с)" >&2

while [ $SECONDS -lt $END_TIME ]; do
    STATUS=$(yc certificate-manager certificate get "$CERT_NAME" --format json 2>/dev/null | jq -r '.status')
    
    if [ "$STATUS" = "ISSUED" ] || [ "$STATUS" = "VALID" ]; then
        echo "Сертификат готов!" >&2
        jq -n --arg status "$STATUS" '{"status":$status, "ready":"true"}'
        exit 0
    elif [ "$STATUS" = "VALIDATING" ]; then
        echo "Сертификат еще проверяется..." >&2
        sleep 10
    else
        echo "Неизвестный статус: $STATUS" >&2
        jq -n --arg status "$STATUS" '{"status":$status, "ready":"false"}'
        exit 1
    fi
done

echo "Timeout! Сертификат не готов за $TIMEOUT секунд" >&2
jq -n '{"status":"TIMEOUT", "ready":"false"}'
exit 1
