FROM nginx:1.29-alpine
COPY static/ /usr/share/nginx/html/
COPY nginx.conf.template /etc/nginx/helion.conf.template
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENV PORT=80 HELION_API_BASE=""
EXPOSE 80
ENTRYPOINT ["/entrypoint.sh"]
