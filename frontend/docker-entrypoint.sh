#!/bin/sh
# Writes the runtime config from env vars, so one image works for any backend URL.
set -eu
API_BASE_URL_ESCAPED=$(printf '%s' "${API_BASE_URL:-http://localhost:8000}" | sed 's/["\\]/\\&/g')
cat > /usr/share/nginx/html/config.js <<EOF
window.__APP_CONFIG__ = { API_BASE_URL: "${API_BASE_URL_ESCAPED}" };
EOF
