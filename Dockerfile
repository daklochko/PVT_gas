FROM python:3.9-slim
FROM python:3.9-slim

WORKDIR /app

COPY . /app
RUN pip install --no-cache-dir -r requirements.txt

CMD ["streamlit", "run", "PVT_gas.py", "--server.port=8501", "--server.enableCORS=false"]
