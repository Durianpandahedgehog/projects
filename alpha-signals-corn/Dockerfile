FROM tensorflow/tensorflow:2.16.1-gpu

WORKDIR /app

RUN pip install --no-cache-dir \
    yfinance==1.3.0 \
    pandas \
    numpy \
    psycopg2-binary \
    sqlalchemy \
    scikit-learn \
    xgboost

COPY . .

CMD ["python", "collect_corn_data.py"]