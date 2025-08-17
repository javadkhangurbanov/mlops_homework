FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install runtime deps with pip (simple & reliable)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code and artifacts
COPY backend/ /app/backend
COPY models/  /app/models
COPY src/     /app/src

# Make /app importable so `import src` and `import backend` work
ENV PYTHONPATH=/app

EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
