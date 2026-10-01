FROM python:3.11-slim
WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
COPY src ./src
COPY checkpoints/minigpt.pt ./checkpoints/minigpt.pt
ENV CHECKPOINT=/app/checkpoints/minigpt.pt PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT}"]
