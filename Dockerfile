FROM python:3.12-slim

# Katalog roboczy
WORKDIR /app

# Wymagania i aplikację
COPY api/requirements.txt requirements.txt
COPY api/app.py app.py
COPY api/model.pkl api/

# Wymagane pakiety
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]
