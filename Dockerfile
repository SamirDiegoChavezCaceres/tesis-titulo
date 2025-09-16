FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencias de sistema para mne
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primero para aprovechar la cache
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto de la app (incluye app.py y model/)
COPY . .

# Exponer puerto
EXPOSE 8000

# Comando para correr la app (ajusta si tu archivo es app.py o main.py)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
