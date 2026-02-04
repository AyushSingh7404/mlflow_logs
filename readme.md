# MLflow Logs API

A lightweight FastAPI service to fetch MLflow experiment data including complete metric history for easy visualization and analysis.

---

## ✅ Key Features

- 🔐 **Basic Authentication** (supports username/password for self-hosted MLflow)
- 📊 **Complete Metric History** (step-by-step points for plotting)
- 🔍 **Flexible querying** by experiment name, experiment ID, or specific run ID
- 📈 **Metric summaries** (min / max / last)
- 🔒 **Optional SSL/TLS bypass** for self-signed certs (use with caution)
- 📝 **Interactive docs** at `/docs`

---

## ⚡ Quick Start

### Requirements

```bash
pip install fastapi uvicorn mlflow pydantic
```

### Run locally

```bash
python mlflow_logs.py
# or with uvicorn for auto-reload
uvicorn mlflow_logs:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: `http://0.0.0.0:8000` and docs at `http://0.0.0.0:8000/docs`

> Note: This service can read defaults from environment variables. If `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, and `MLFLOW_TRACKING_PASSWORD` are set in the environment, you may omit `tracking_uri`, `username`, and `password` in the request body (you can send an empty JSON `{}` to use environment defaults). If these are not set, provide them in the request.

---

## Endpoints Overview

- POST `/health` — Check MLflow accessibility
- POST `/mlflow/experiments` — List experiments
- POST `/mlflow/runs` — Fetch runs and full metric history

---

## Example Requests & Optional Responses 🔍

### 1. List experiments (using environment defaults)

Request (empty body is valid when env vars set):

```bash
curl -X POST http://0.0.0.0:8000/mlflow/experiments \
  -H "Content-Type: application/json" \
  -d '{}'
```

Possible Response (success):

```json
{
  "success": true,
  "total_experiments": 3,
  "experiments": [
    {"experiment_id": "1", "name": "391-818", "artifact_location": "s3://bucket/path", "lifecycle_stage": "active"}
  ]
}
```

If the server cannot reach MLflow or auth fails, you'll get an HTTP 500 or a descriptive error.

---

### 2. Fetch runs for an experiment (with an explicit experiment name)

Request:

```bash
curl -X POST http://0.0.0.0:8000/mlflow/runs \
  -H "Content-Type: application/json" \
  -d '{
    "experiment_name": "391-818",
    "limit": 5
  }'
```

Possible Response (success):

```json
{
  "success": true,
  "experiment_name": "391-818",
  "experiment_id": "2",
  "total_runs": 2,
  "runs": [
    {
      "run_id": "abc123",
      "run_name": "training_run_1",
      "status": "FINISHED",
      "duration_seconds": 3600.0,
      "parameters": {"lr": "0.001"},
      "metrics_summary": {"val_accuracy": 0.89},
      "metrics_history": [
        {"metric_name":"val_accuracy","history":[{"step":0,"value":0.8,"timestamp":1700000000000}],"total_points":1,"min_value":0.8,"max_value":0.8,"last_value":0.8}
      ]
    }
  ],
  "message": "Fetched 2 run(s) with 3 total metric histories"
}
```

If the experiment is not found, you'll receive a 404 response with an explanatory message.

---

## Request Fields (high level)

- `tracking_uri` (string) — MLflow tracking server URL (optional if `MLFLOW_TRACKING_URI` env var is set)
- `username` (string) — MLflow username (optional if `MLFLOW_TRACKING_USERNAME` env var is set)
- `password` (string) — MLflow password (optional if `MLFLOW_TRACKING_PASSWORD` env var is set)
- `insecure_tls` (bool, default true) — Disable SSL verification for self-signed certs
- `http_timeout` (int, default 15) — HTTP request timeout in seconds
- `experiment_name` / `experiment_id` — experiment selector
- `run_id` — fetch a single run
- `limit` (int, default 10) — max runs to return
- `include_metric_history` (bool, default true) — include full metric histories

> Tip: For quick local testing set the three env vars and call the endpoints with `{}` bodies.

---

## Interactive Docs

Open `http://0.0.0.0:8000/docs` for Swagger UI to try endpoints and view request/response schemas.

---

## Docker (optional)

Dockerfile should copy `mlflow_logs.py` and expose port `8000`.

Example Dockerfile snippet:

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY mlflow_logs.py .
EXPOSE 8000
CMD ["python", "mlflow_logs.py"]
```

Build and run:

```bash
docker build -t mlflow-api .
docker run -p 8000:8000 mlflow-api
```

---

## Troubleshooting & Notes

- If you rely on environment defaults but still get validation errors, check that `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, and `MLFLOW_TRACKING_PASSWORD` are set in the process environment that starts the API.
- `insecure_tls: true` is convenient for self-signed certs, but **do not** use it in production environments.

---
