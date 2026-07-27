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
# per URL.

WORKDIR /app

# One apt pass for curl and Chromium's shared libraries.
#
# Deliberately NOT `playwright install --with-deps`. That flag installs Playwright's
# full OS dependency set, which on Debian pulls xvfb, xserver-common, x11-xkb-utils and
# the xfonts-* packages -- an entire X server for a browser we only ever launch headless
# to read cookies. It is also what broke the build twice on a 4-CPU/8GB daemon: once
# OOM-killed outright (exit 137), once taking BuildKit down with it mid-transaction.
#
# Below is what headless Chromium actually links against, using Debian trixie's t64
# package names. Fonts are omitted on purpose: nothing here renders text for a human,
# it navigates one page and reads the cookie jar. If Chromium ever fails to start with
# a `error while loading shared libraries` message, the missing name is in that error --
# add it here rather than reaching back for --with-deps.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        libnss3 libnspr4 libdbus-1-3 \
        libatk1.0-0t64 libatk-bridge2.0-0t64 libatspi2.0-0t64 \
        libcups2t64 libdrm2 libgbm1 libasound2t64 \
        libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 \
        libxshmfence1 libx11-6 libxcb1 libxext6 \
        libpango-1.0-0 libcairo2 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Browser binary only -- its system libraries came from the apt layer above.
RUN pip install --no-cache-dir -r requirements.txt \
    && playwright install chromium

COPY . .

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
