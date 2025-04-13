FROM python:3.12-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml uv.lock* ./

RUN uv venv

RUN uv sync

COPY ./src /app/src

EXPOSE 8000

CMD ["python", "-m", "src.main"]
