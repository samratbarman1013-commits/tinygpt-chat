FROM python:3.12-slim

WORKDIR /app

# CPU-only torch first (big layer, cached well)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV HOST=0.0.0.0 PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "uvicorn app:app --host ${HOST} --port ${PORT}"]
