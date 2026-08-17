FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY prompts ./prompts
COPY data ./data

RUN pip install --no-cache-dir .

ENTRYPOINT ["model-eval"]
CMD ["--prompt", "prompts/v1.yaml", "--dataset", "data/golden/v1.json"]
