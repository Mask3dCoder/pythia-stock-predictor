FROM python:3.11-slim

LABEL maintainer="Pythia Team <pythia@example.com>"
LABEL description="Pythia Stock Predictor - Bloomberg Terminal for Everyone"

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./

RUN pip install --no-cache-dir .

RUN pip install --no-cache-dir "pythia-stock-predictor[ml,visualization]"

COPY . .

ENV PYTHONUNBUFFERED=1
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
