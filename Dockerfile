FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY google_ads_server.py ./
COPY web_server.py ./

RUN pip install --no-cache-dir ".[web]"

ENV PORT=8080
EXPOSE 8080

CMD ["python", "-c", "from web_server import run_web_server; run_web_server()"]
