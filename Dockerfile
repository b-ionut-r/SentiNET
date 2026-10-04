# syntax=docker/dockerfile:1.7
# SentiNET — one image serving the API and the built web app on :8000.
#   docker build -t sentinet .
#   docker run -p 8000:8000 -v sentinet-data:/data --env-file backend/.env sentinet

# ---- 1. Build the web app ----------------------------------------------------------
FROM node:22-alpine AS web
WORKDIR /web
# Playwright is a dev-only dependency (visual QA); never fetch browsers here.
ENV PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ---- 2. Python runtime ------------------------------------------------------------
FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DATABASE_PATH=/data/sentinet.db \
    SENTINET_FRONTEND_DIST=/app/frontend/dist

WORKDIR /app/backend
COPY backend/requirements.txt ./
RUN pip install -r requirements.txt

COPY backend/app ./app
COPY --from=web /web/dist /app/frontend/dist

# Unprivileged user with a real home (yfinance keeps a small cache in ~/.cache).
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin sentinet \
    && mkdir -p /data \
    && chown sentinet:sentinet /data
USER sentinet
VOLUME ["/data"]

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
