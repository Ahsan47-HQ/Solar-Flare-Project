#!/bin/sh
set -eu
if [ -n "$HELION_API_BASE" ] && ! printf '%s' "$HELION_API_BASE" | grep -Eq '^https?://[A-Za-z0-9.-]+(:[0-9]{1,5})?$'; then
    echo 'HELION_API_BASE must be an HTTP(S) origin without a path.' >&2
    exit 1
fi
envsubst '${PORT}' < /etc/nginx/helion.conf.template > /etc/nginx/conf.d/default.conf
printf 'window.HELION_API_BASE = "%s";\n' "$HELION_API_BASE" > /usr/share/nginx/html/config.js
exec nginx -g 'daemon off;'
