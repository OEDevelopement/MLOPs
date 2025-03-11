FROM grafana/grafana:latest

# Set environment variables
ENV GF_SECURITY_ADMIN_USER=admin
ENV GF_SECURITY_ADMIN_PASSWORD=admin
ENV GF_USERS_ALLOW_SIGN_UP=false

# Copy provisioning and dashboards
COPY provisioning /etc/grafana/provisioning
COPY dashboards /var/lib/grafana/dashboards

EXPOSE 3000