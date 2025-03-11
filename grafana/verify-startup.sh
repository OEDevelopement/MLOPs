#!/bin/bash
# Save this as grafana/verify-startup.sh

# Make the script executable
# chmod +x grafana/verify-startup.sh

echo "=== Grafana Startup Verification ==="

# Check if Grafana is running
echo "Checking if Grafana is running..."
if curl -s http://grafana:3000/api/health | grep -q "ok"; then
  echo "✅ Grafana is running"
else
  echo "❌ Grafana is not running properly"
fi

# Check if Prometheus datasource is configured
echo "Checking Prometheus datasource..."
if curl -s -u admin:admin http://grafana:3000/api/datasources | grep -q "prometheus"; then
  echo "✅ Prometheus datasource is configured"
else
  echo "❌ Prometheus datasource is not configured properly"
fi

# Check if dashboards are loaded
echo "Checking dashboards..."
if curl -s -u admin:admin http://grafana:3000/api/search | grep -q "dashboard"; then
  echo "✅ Dashboards are loaded"
else
  echo "❌ Dashboards are not loaded properly"
fi

echo "=== Verification Complete ==="