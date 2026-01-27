# MLflow Logs API

A FastAPI-based REST API to fetch MLflow experiment data with complete metric history for visualization and analysis.

## Features

- 🔐 **Basic Authentication** - Username/password authentication for self-hosted MLflow
- 📊 **Complete Metric History** - Fetch step-by-step metric data for plotting
- 🔍 **Flexible Queries** - Search by experiment name, ID, or run ID
- 📈 **Statistical Summaries** - Get min/max/last values for each metric
- 🔒 **SSL Bypass** - Support for self-signed certificates
- 📝 **Auto-generated Docs** - Interactive API documentation at `/docs`

## Installation

### Requirements

```bash
pip install fastapi uvicorn mlflow pydantic
```

### Quick Start

1. Run the API:
```bash
python mlflow_logs_api.py
```

The API will be available at `http://localhost:8001`

## API Endpoints

### 1. Health Check
**POST** `/health`

Check if your MLflow server is accessible.

**Request:**
```json
{
  "tracking_uri": "https://mlflow.example.com/mlflow-api",
  "username": "your_username",
  "password": "your_password",
  "insecure_tls": true
}
```

**Response:**
```json
{
  "status": "healthy",
  "mlflow_accessible": true,
  "experiments_found": 5,
  "authentication": "basic_auth",
  "ssl_verification": false
}
```

---

### 2. List Experiments
**POST** `/mlflow/experiments`

Get all available experiments from your MLflow server.

**Request:**
```json
{
  "tracking_uri": "https://mlflow.example.com/mlflow-api",
  "username": "your_username",
  "password": "your_password",
  "insecure_tls": true
}
```

**Response:**
```json
{
  "success": true,
  "total_experiments": 3,
  "experiments": [
    {
      "experiment_id": "1",
      "name": "my-experiment",
      "artifact_location": "s3://bucket/path",
      "lifecycle_stage": "active"
    }
  ]
}
```

---

### 3. Fetch Runs
**POST** `/mlflow/runs`

Fetch runs from an experiment with complete metric history.

**Request:**
```json
{
  "tracking_uri": "https://mlflow.example.com/mlflow-api",
  "username": "your_username",
  "password": "your_password",
  "insecure_tls": true,
  "experiment_name": "my-experiment",
  "limit": 10,
  "include_metric_history": true
}
```

**Response:**
```json
{
  "success": true,
  "experiment_name": "my-experiment",
  "total_runs": 2,
  "runs": [
    {
      "run_id": "abc123",
      "run_name": "training_run_1",
      "status": "FINISHED",
      "duration_seconds": 3600.0,
      "parameters": {
        "learning_rate": "0.001",
        "batch_size": "16"
      },
      "metrics_summary": {
        "train_loss": 0.234,
        "val_accuracy": 0.892
      },
      "metrics_history": [
        {
          "metric_name": "train_loss",
          "history": [
            {"step": 0, "value": 0.5, "timestamp": 1704067200000},
            {"step": 1, "value": 0.4, "timestamp": 1704068400000},
            {"step": 2, "value": 0.234, "timestamp": 1704069600000}
          ],
          "total_points": 3,
          "min_value": 0.234,
          "max_value": 0.5,
          "last_value": 0.234
        }
      ]
    }
  ],
  "message": "Fetched 2 run(s) with 4 total metric histories"
}
```

## Configuration Parameters

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `tracking_uri` | ✅ Yes | - | MLflow server URL |
| `username` | ✅ Yes | - | MLflow username |
| `password` | ✅ Yes | - | MLflow password |
| `insecure_tls` | No | `true` | Disable SSL verification for self-signed certs |
| `http_timeout` | No | `15` | HTTP request timeout in seconds |
| `experiment_name` | No | - | Filter by experiment name |
| `experiment_id` | No | - | Filter by experiment ID |
| `run_id` | No | - | Fetch specific run by ID |
| `limit` | No | `10` | Maximum number of runs to fetch |
| `include_metric_history` | No | `true` | Include complete step-by-step metrics |

## Usage Examples

### Python Example

```python
import requests

# Configuration
config = {
    "tracking_uri": "https://mlflow.example.com/mlflow-api",
    "username": "mlflow",
    "password": "your_password",
    "insecure_tls": True,
    "experiment_name": "my-experiment",
    "limit": 5,
    "include_metric_history": True
}

# Fetch runs
response = requests.post(
    "http://localhost:8001/mlflow/runs",
    json=config
)

data = response.json()

# Access metric history
for run in data['runs']:
    print(f"Run: {run['run_name']}")
    for metric in run['metrics_history']:
        print(f"  {metric['metric_name']}: {metric['total_points']} points")
        steps = [p['step'] for p in metric['history']]
        values = [p['value'] for p in metric['history']]
        # Now you can plot steps vs values
```

### cURL Example

```bash
curl -X POST "http://localhost:8001/mlflow/runs" \
  -H "Content-Type: application/json" \
  -d '{
    "tracking_uri": "https://mlflow.example.com/mlflow-api",
    "username": "mlflow",
    "password": "your_password",
    "insecure_tls": true,
    "experiment_name": "my-experiment"
  }'
```

### JavaScript/Fetch Example

```javascript
const config = {
  tracking_uri: "https://mlflow.example.com/mlflow-api",
  username: "mlflow",
  password: "your_password",
  insecure_tls: true,
  experiment_name: "my-experiment",
  limit: 10
};

fetch("http://localhost:8001/mlflow/runs", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(config)
})
  .then(res => res.json())
  .then(data => {
    console.log(`Found ${data.total_runs} runs`);
    data.runs.forEach(run => {
      console.log(`Run: ${run.run_name}`);
    });
  });
```

## Interactive Documentation

Once the API is running, visit:

```
http://localhost:8001/docs
```

This provides an interactive Swagger UI where you can:
- Test all endpoints
- See request/response schemas
- Try queries directly in the browser

## Response Structure

### Run Data Object

Each run contains:

- **`run_id`** - Unique run identifier
- **`run_name`** - Human-readable run name
- **`status`** - Run status (FINISHED, RUNNING, FAILED, etc.)
- **`duration_seconds`** - Total run duration
- **`parameters`** - All hyperparameters logged during training
- **`metrics_summary`** - Latest value for each metric
- **`metrics_history`** - Complete step-by-step history for each metric

### Metric History Object

Each metric history contains:

- **`metric_name`** - Name of the metric
- **`history`** - Array of {step, value, timestamp} objects
- **`total_points`** - Number of data points
- **`min_value`** - Minimum value across all steps
- **`max_value`** - Maximum value across all steps
- **`last_value`** - Most recent value

## Use Cases

### 1. Training Dashboard
Fetch metrics in real-time to display training progress on a dashboard.

### 2. Model Comparison
Compare hyperparameters and metrics across multiple training runs.

### 3. Automated Reporting
Generate reports with training metrics and visualizations.

### 4. Data Analysis
Export metric data to CSV/Excel for offline analysis.

### 5. Integration
Integrate MLflow data into your existing tools and workflows.

## Security Notes

### SSL/TLS

The `insecure_tls` parameter disables SSL certificate verification. This is useful for:
- Self-signed certificates
- Internal development environments
- Testing

**For production:**
- Use proper SSL certificates
- Set `insecure_tls: false`
- Ensure certificates are in your system's trust store

### Authentication

All requests require username and password. Store credentials securely:

```python
# ❌ Don't hardcode credentials
config = {"username": "mlflow", "password": "mypassword"}

# ✅ Use environment variables
import os
config = {
    "username": os.getenv("MLFLOW_USERNAME"),
    "password": os.getenv("MLFLOW_PASSWORD")
}
```

## Troubleshooting

### Connection Issues

**Error:** "SSL: CERTIFICATE_VERIFY_FAILED"
```json
{
  "insecure_tls": true
}
```

**Error:** "Connection timeout"
```json
{
  "http_timeout": 30
}
```

### Authentication Issues

**Error:** "Authentication failed"
- Verify username and password
- Check MLflow server authentication settings

### Experiment Not Found

**Error:** "Experiment not found"
- List all experiments first: `POST /mlflow/experiments`
- Use exact experiment name from the list

## Development

### Run in Development Mode

```bash
uvicorn mlflow_logs_api:app --reload --host 0.0.0.0 --port 8001
```

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY mlflow_logs_api.py .

EXPOSE 8001

CMD ["python", "mlflow_logs_api.py"]
```

Build and run:
```bash
docker build -t mlflow-api .
docker run -p 8001:8001 mlflow-api
```
