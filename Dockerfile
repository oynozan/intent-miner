FROM python:3.12-slim

# No Playwright, no browser deps, no Xvfb. Both "heavy" platforms turned out to be
# plain HTTP: LinkedIn publishes /posts/ for Googlebot with the body in ld+json, and
# Quora ships its content in inline script payloads. The browser stack the original
# spec called for would have been the largest thing in this image and bought nothing.

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
