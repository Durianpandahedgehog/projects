@echo off
cd C:\Users\rolan\Downloads\corn-predictor
docker exec corn_app python dashboard.py
docker cp corn_app:/app/dashboard.html ./dashboard.html
start dashboard.html