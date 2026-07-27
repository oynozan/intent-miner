FROM python:3.12-slim

# A browser, but only just. LinkedIn and Quora are both plain HTTP -- LinkedIn publishes
# /posts/ for Googlebot with the body in ld+json, Quora ships content in inline script
# payloads -- so the full browser stack the original spec called for still buys nothing
# there, and Quora is now SERP-only anyway.
#
# Chromium is here for one job: minting Reddit's cookie jar. Reddit's .json is 403
# without cookies and 200 with them, and the cookies are written by page JavaScript that
# neither curl_cffi nor obscura executes (both return a lone `edgebucket`; Chromium
# returns 12). That runs about once per 2h TTL, shared across workers via Redis -- never
# per URL. Only chromium is installed, not --with-deps' full browser set.

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && playwright install --with-deps chromium \
    && rm -rf /var/lib/apt/lists/*

COPY . .

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
