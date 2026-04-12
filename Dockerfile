FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
RUN pip install --no-cache-dir .

EXPOSE 7860

CMD ["sh", "-c", "if [ \"$SPACE_ID\" ]; then uvicorn server.app:app --host 0.0.0.0 --port 7860; else python inference.py; fi"]
