FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["python", "-c", "import os, subprocess, sys; target = [sys.executable, '-m', 'server.app'] if os.getenv('PORT') else [sys.executable, 'inference.py']; raise SystemExit(subprocess.call(target))"]
