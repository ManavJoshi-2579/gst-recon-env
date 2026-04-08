FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir fastapi "uvicorn[standard]" pydantic numpy

COPY . /app

EXPOSE 8000

CMD ["python", "-m", "server.app"]
