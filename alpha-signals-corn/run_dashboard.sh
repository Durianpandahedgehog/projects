#!/bin/bash
echo "Generating today's corn signal..."
docker exec corn_app python dashboard.py
docker cp corn_app:/app/dashboard.html ./dashboard.html
open dashboard.html