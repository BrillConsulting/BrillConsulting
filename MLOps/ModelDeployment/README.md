# Model Deployment

Production-ready model serving with FastAPI and Docker.

## Features

- REST API with FastAPI
- Input validation with Pydantic
- Batch predictions
- Health checks and metrics
- Docker containerization
- Auto-generated API docs

## Usage

```bash
# Start server
uvicorn model_server:app --reload

# Make prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [[1.0, 2.0, 3.0]]}'
```

## Docker

```dockerfile
FROM python:3.9
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["uvicorn", "model_server:app", "--host", "0.0.0.0"]
```

## Demo

```bash
python model_server.py
```
