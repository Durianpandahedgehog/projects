#!/bin/bash
echo "============================================"
echo " CORN PREDICTOR - First Time Setup"
echo "============================================"
echo ""

echo "[1/6] Starting Docker containers..."
docker compose up --build -d
sleep 15

echo ""
echo "[2/6] Waiting for database to be ready..."
sleep 10

echo ""
echo "[3/6] Loading 20 years of historical data..."
echo "This may take a few minutes..."
docker exec corn_app pip install tzdata
docker exec corn_app python collect_corn_data.py

echo ""
echo "[4/6] Training the ensemble model (XGBoost + LSTM)..."
echo "This will take 5-10 minutes..."
docker exec corn_app python ml_model_ensemble.py

echo ""
echo "[5/6] Generating today's signal..."
docker exec corn_app python dashboard.py

echo ""
echo "[6/6] Copying dashboard to local folder..."
docker cp corn_app:/app/dashboard.html ./dashboard.html

echo ""
echo "============================================"
echo " Setup complete! Opening dashboard..."
echo "============================================"
open dashboard.html