# """
# Minimal FastAPI endpoint to fetch MLflow experiment data.
# All configuration comes from request parameters only.
# """

# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel, Field
# from typing import Optional, List, Dict, Any
# from mlflow.tracking import MlflowClient

# app = FastAPI(
#     title="MLflow Data API - Simplified",
#     description="Fetch MLflow experiment data",
#     version="2.0.0"
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ===== Models =====
# class MLflowRequest(BaseModel):
#     tracking_uri: str = Field(
#         default="./mlruns",
#         description="MLflow tracking server URI (e.g., 'http://localhost:5000' or './mlruns')"
#     )
#     databricks_token: Optional[str] = Field(
#         None,
#         description="Databricks access token (required for Databricks MLflow)"
#     )
#     experiment_name: Optional[str] = Field(
#         None, 
#         description="Experiment name to fetch (e.g., 'my-experiment')"
#     )
#     experiment_id: Optional[str] = Field(
#         None, 
#         description="Experiment ID to fetch (e.g., '0', '1', '2')"
#     )
#     run_id: Optional[str] = Field(
#         None, 
#         description="Specific run ID to fetch"
#     )
#     limit: Optional[int] = Field(
#         10, 
#         description="Maximum number of runs to fetch"
#     )

# class RunData(BaseModel):
#     run_id: str
#     run_name: str
#     experiment_id: str
#     status: str
#     start_time: Optional[int] = None
#     end_time: Optional[int] = None
#     parameters: Dict[str, Any] = {}
#     metrics: Dict[str, float] = {}
#     tags: Dict[str, str] = {}

# class MLflowResponse(BaseModel):
#     success: bool
#     experiment_name: Optional[str] = None
#     experiment_id: Optional[str] = None
#     total_runs: int
#     runs: List[RunData]
#     message: Optional[str] = None

# # ===== Helper Functions =====
# def get_mlflow_client(tracking_uri: str, databricks_token: Optional[str] = None):
#     """Initialize MLflow client with tracking URI and optional Databricks token."""
#     import os
    
#     # Set Databricks token if provided (multiple env vars for compatibility)
#     if databricks_token:
#         os.environ["DATABRICKS_TOKEN"] = databricks_token
#         os.environ["DATABRICKS_HOST"] = tracking_uri
#         # Also set as MLFLOW env var for authentication
#         os.environ["MLFLOW_TRACKING_TOKEN"] = databricks_token
    
#     return MlflowClient(tracking_uri=tracking_uri)

# def process_single_run(client: MlflowClient, run_id: str) -> RunData:
#     """Fetch and process a single MLflow run."""
#     run = client.get_run(run_id)
    
#     return RunData(
#         run_id=run.info.run_id,
#         run_name=run.data.tags.get("mlflow.runName", run.info.run_id[:8]),
#         experiment_id=run.info.experiment_id,
#         status=run.info.status,
#         start_time=run.info.start_time,
#         end_time=run.info.end_time,
#         parameters=run.data.params,
#         metrics=run.data.metrics,
#         tags=run.data.tags
#     )

# # ===== API Endpoints =====
# @app.get("/")
# async def root():
#     return {
#         "status": "online",
#         "service": "MLflow Data API - Simplified",
#         "version": "2.0.0",
#         "endpoints": {
#             "fetch_runs": "/mlflow/runs (POST)",
#             "health": "/health (POST)",
#             "docs": "/docs"
#         },
#         "note": "All configuration is provided via request body"
#     }

# @app.post("/mlflow/runs", response_model=MLflowResponse)
# async def fetch_mlflow_runs(request: MLflowRequest):
#     """
#     Fetch MLflow runs from an experiment.
    
#     Examples:
#     - Local: {"tracking_uri": "./mlruns", "experiment_name": "my-experiment"}
#     - Remote: {"tracking_uri": "http://localhost:5000", "experiment_id": "0"}
#     - Databricks: {"tracking_uri": "https://dbc-xxx.cloud.databricks.com", "databricks_token": "dapi...", "experiment_id": "123"}
#     - Single run: {"tracking_uri": "http://localhost:5000", "run_id": "abc123"}
#     """
#     try:
#         # Initialize client with provided tracking URI and token
#         client = get_mlflow_client(request.tracking_uri, request.databricks_token)
        
#         # Case 1: Fetch single run by ID
#         if request.run_id:
#             try:
#                 run_data = process_single_run(client, request.run_id)
#                 run = client.get_run(request.run_id)
#                 experiment = client.get_experiment(run.info.experiment_id)
                
#                 return MLflowResponse(
#                     success=True,
#                     experiment_name=experiment.name,
#                     experiment_id=experiment.experiment_id,
#                     total_runs=1,
#                     runs=[run_data],
#                     message=f"Successfully fetched run: {request.run_id}"
#                 )
#             except Exception as e:
#                 raise HTTPException(
#                     status_code=404,
#                     detail=f"Run '{request.run_id}' not found: {str(e)}"
#                 )
        
#         # Case 2: Fetch runs from an experiment
#         # Get experiment by name or ID
#         if request.experiment_name:
#             experiment = client.get_experiment_by_name(request.experiment_name)
#             if not experiment:
#                 raise HTTPException(
#                     status_code=404,
#                     detail=f"Experiment '{request.experiment_name}' not found"
#                 )
#             experiment_id = experiment.experiment_id
#             experiment_name = experiment.name
#         elif request.experiment_id:
#             experiment = client.get_experiment(request.experiment_id)
#             if not experiment:
#                 raise HTTPException(
#                     status_code=404,
#                     detail=f"Experiment ID '{request.experiment_id}' not found"
#                 )
#             experiment_id = experiment.experiment_id
#             experiment_name = experiment.name
#         else:
#             # Default: get first experiment (usually "Default" experiment with ID "0")
#             experiments = client.search_experiments()
#             if not experiments:
#                 raise HTTPException(
#                     status_code=404,
#                     detail="No experiments found in MLflow tracking server"
#                 )
#             experiment = experiments[0]
#             experiment_id = experiment.experiment_id
#             experiment_name = experiment.name
        
#         # Search runs in the experiment
#         runs = client.search_runs(
#             experiment_ids=[experiment_id],
#             max_results=request.limit or 10
#         )
        
#         if not runs:
#             return MLflowResponse(
#                 success=True,
#                 experiment_name=experiment_name,
#                 experiment_id=experiment_id,
#                 total_runs=0,
#                 runs=[],
#                 message=f"No runs found in experiment '{experiment_name}'"
#             )
        
#         # Process all runs
#         runs_data = [
#             process_single_run(client, run.info.run_id) 
#             for run in runs
#         ]
        
#         return MLflowResponse(
#             success=True,
#             experiment_name=experiment_name,
#             experiment_id=experiment_id,
#             total_runs=len(runs_data),
#             runs=runs_data,
#             message=f"Successfully fetched {len(runs_data)} run(s) from '{experiment_name}'"
#         )
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Error fetching MLflow data: {str(e)}"
#         )

# @app.post("/health")
# async def health_check(tracking_uri: str = "./mlruns", databricks_token: Optional[str] = None):
#     """
#     Health check endpoint.
#     Provide tracking_uri as query parameter: /health?tracking_uri=http://localhost:5000
#     For Databricks, also provide databricks_token
#     """
#     try:
#         client = get_mlflow_client(tracking_uri, databricks_token)
#         experiments = client.search_experiments(max_results=1)
#         mlflow_accessible = True
#         experiment_count = len(experiments)
#     except Exception as e:
#         mlflow_accessible = False
#         experiment_count = 0
    
#     return {
#         "status": "healthy",
#         "mlflow_accessible": mlflow_accessible,
#         "tracking_uri": tracking_uri,
#         "experiments_found": experiment_count
#     }

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8001)






"""
Enhanced MLflow API to fetch complete metric history (step-by-step data).
Includes all metrics with their full history for visualization.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from mlflow.tracking import MlflowClient

app = FastAPI(
    title="MLflow Data API - Enhanced",
    description="Fetch complete MLflow experiment data with metric history",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Models =====
class MLflowRequest(BaseModel):
    tracking_uri: str = Field(
        default="./mlruns",
        description="MLflow tracking server URI (e.g., 'http://localhost:5000' or './mlruns')"
    )
    databricks_token: Optional[str] = Field(
        None,
        description="Databricks access token (required for Databricks MLflow)"
    )
    experiment_name: Optional[str] = Field(
        None, 
        description="Experiment name to fetch (e.g., 'my-experiment')"
    )
    experiment_id: Optional[str] = Field(
        None, 
        description="Experiment ID to fetch (e.g., '0', '1', '2')"
    )
    run_id: Optional[str] = Field(
        None, 
        description="Specific run ID to fetch"
    )
    limit: Optional[int] = Field(
        10, 
        description="Maximum number of runs to fetch"
    )
    include_metric_history: bool = Field(
        True,
        description="Include complete step-by-step metric history"
    )

class MetricPoint(BaseModel):
    """Single metric data point with step and timestamp"""
    step: int
    value: float
    timestamp: int

class MetricHistory(BaseModel):
    """Complete history of a metric across all steps"""
    metric_name: str
    history: List[MetricPoint]
    total_points: int
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    last_value: Optional[float] = None

class RunData(BaseModel):
    run_id: str
    run_name: str
    experiment_id: str
    status: str
    start_time: Optional[int] = None
    end_time: Optional[int] = None
    duration_seconds: Optional[float] = None
    parameters: Dict[str, Any] = {}
    metrics_summary: Dict[str, float] = {}  # Latest values
    metrics_history: List[MetricHistory] = []  # Complete step-by-step history
    tags: Dict[str, str] = {}
    artifact_uri: Optional[str] = None

class MLflowResponse(BaseModel):
    success: bool
    experiment_name: Optional[str] = None
    experiment_id: Optional[str] = None
    total_runs: int
    runs: List[RunData]
    message: Optional[str] = None

# ===== Helper Functions =====
def get_mlflow_client(tracking_uri: str, databricks_token: Optional[str] = None):
    """Initialize MLflow client with tracking URI and optional Databricks token."""
    import os
    
    # Set Databricks token if provided
    if databricks_token:
        os.environ["DATABRICKS_TOKEN"] = databricks_token
        os.environ["DATABRICKS_HOST"] = tracking_uri
        os.environ["MLFLOW_TRACKING_TOKEN"] = databricks_token
    
    return MlflowClient(tracking_uri=tracking_uri)

def get_metric_history(client: MlflowClient, run_id: str, metric_key: str) -> MetricHistory:
    """Fetch complete history of a single metric."""
    try:
        history = client.get_metric_history(run_id, metric_key)
        
        if not history:
            return MetricHistory(
                metric_name=metric_key,
                history=[],
                total_points=0
            )
        
        # Convert to list of MetricPoints
        metric_points = [
            MetricPoint(
                step=int(m.step),
                value=float(m.value),
                timestamp=int(m.timestamp)
            )
            for m in history
        ]
        
        # Sort by step
        metric_points.sort(key=lambda x: x.step)
        
        # Calculate statistics
        values = [p.value for p in metric_points]
        
        return MetricHistory(
            metric_name=metric_key,
            history=metric_points,
            total_points=len(metric_points),
            min_value=min(values) if values else None,
            max_value=max(values) if values else None,
            last_value=values[-1] if values else None
        )
    except Exception as e:
        print(f"Warning: Could not fetch history for metric '{metric_key}': {e}")
        return MetricHistory(
            metric_name=metric_key,
            history=[],
            total_points=0
        )

def process_single_run(client: MlflowClient, run_id: str, include_history: bool = True) -> RunData:
    """Fetch and process a single MLflow run with complete metric history."""
    run = client.get_run(run_id)
    
    # Calculate duration
    duration = None
    if run.info.end_time and run.info.start_time:
        duration = (run.info.end_time - run.info.start_time) / 1000.0  # Convert ms to seconds
    
    # Get metric history for all metrics
    metrics_history = []
    if include_history and run.data.metrics:
        for metric_key in run.data.metrics.keys():
            history = get_metric_history(client, run_id, metric_key)
            metrics_history.append(history)
    
    return RunData(
        run_id=run.info.run_id,
        run_name=run.data.tags.get("mlflow.runName", run.info.run_id[:8]),
        experiment_id=run.info.experiment_id,
        status=run.info.status,
        start_time=run.info.start_time,
        end_time=run.info.end_time,
        duration_seconds=duration,
        parameters=run.data.params,
        metrics_summary=run.data.metrics,  # Latest values only
        metrics_history=metrics_history,  # Complete history
        tags=run.data.tags,
        artifact_uri=run.info.artifact_uri
    )

# ===== API Endpoints =====
@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "MLflow Data API - Enhanced with Metric History",
        "version": "3.0.0",
        "endpoints": {
            "fetch_runs": "/mlflow/runs (POST)",
            "fetch_run_detailed": "/mlflow/run/{run_id} (GET)",
            "list_experiments": "/mlflow/experiments (GET)",
            "health": "/health (POST)",
            "docs": "/docs"
        },
        "features": [
            "Complete step-by-step metric history",
            "System metrics with timestamps",
            "Training and evaluation metrics",
            "Statistical summaries (min/max/last)",
            "Ready for visualization"
        ]
    }

@app.post("/mlflow/runs", response_model=MLflowResponse)
async def fetch_mlflow_runs(request: MLflowRequest):
    """
    Fetch MLflow runs with complete metric history.
    
    Examples:
    - Local: {"tracking_uri": "./mlruns", "experiment_name": "my-experiment"}
    - Remote: {"tracking_uri": "http://localhost:5000", "experiment_id": "0"}
    - Databricks: {"tracking_uri": "https://dbc-xxx.cloud.databricks.com", "databricks_token": "dapi...", "experiment_id": "123"}
    - Single run: {"tracking_uri": "http://localhost:5000", "run_id": "abc123", "include_metric_history": true}
    """
    try:
        # Initialize client
        client = get_mlflow_client(request.tracking_uri, request.databricks_token)
        
        # Case 1: Fetch single run by ID
        if request.run_id:
            try:
                run_data = process_single_run(client, request.run_id, request.include_metric_history)
                run = client.get_run(request.run_id)
                experiment = client.get_experiment(run.info.experiment_id)
                
                return MLflowResponse(
                    success=True,
                    experiment_name=experiment.name,
                    experiment_id=experiment.experiment_id,
                    total_runs=1,
                    runs=[run_data],
                    message=f"Successfully fetched run with {len(run_data.metrics_history)} metrics"
                )
            except Exception as e:
                raise HTTPException(
                    status_code=404,
                    detail=f"Run '{request.run_id}' not found: {str(e)}"
                )
        
        # Case 2: Fetch runs from an experiment
        if request.experiment_name:
            experiment = client.get_experiment_by_name(request.experiment_name)
            if not experiment:
                raise HTTPException(
                    status_code=404,
                    detail=f"Experiment '{request.experiment_name}' not found"
                )
            experiment_id = experiment.experiment_id
            experiment_name = experiment.name
        elif request.experiment_id:
            experiment = client.get_experiment(request.experiment_id)
            if not experiment:
                raise HTTPException(
                    status_code=404,
                    detail=f"Experiment ID '{request.experiment_id}' not found"
                )
            experiment_id = experiment.experiment_id
            experiment_name = experiment.name
        else:
            # Default: get first experiment
            experiments = client.search_experiments()
            if not experiments:
                raise HTTPException(
                    status_code=404,
                    detail="No experiments found in MLflow tracking server"
                )
            experiment = experiments[0]
            experiment_id = experiment.experiment_id
            experiment_name = experiment.name
        
        # Search runs
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            max_results=request.limit or 10
        )
        
        if not runs:
            return MLflowResponse(
                success=True,
                experiment_name=experiment_name,
                experiment_id=experiment_id,
                total_runs=0,
                runs=[],
                message=f"No runs found in experiment '{experiment_name}'"
            )
        
        # Process all runs
        runs_data = [
            process_single_run(client, run.info.run_id, request.include_metric_history) 
            for run in runs
        ]
        
        total_metrics = sum(len(r.metrics_history) for r in runs_data)
        
        return MLflowResponse(
            success=True,
            experiment_name=experiment_name,
            experiment_id=experiment_id,
            total_runs=len(runs_data),
            runs=runs_data,
            message=f"Fetched {len(runs_data)} run(s) with {total_metrics} total metric histories"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching MLflow data: {str(e)}"
        )

@app.get("/mlflow/run/{run_id}")
async def fetch_single_run(
    run_id: str,
    tracking_uri: str = "./mlruns",
    databricks_token: Optional[str] = None,
    include_metric_history: bool = True
):
    """
    Fetch a single run with complete metric history.
    Query parameters: tracking_uri, databricks_token, include_metric_history
    """
    try:
        client = get_mlflow_client(tracking_uri, databricks_token)
        run_data = process_single_run(client, run_id, include_metric_history)
        run = client.get_run(run_id)
        experiment = client.get_experiment(run.info.experiment_id)
        
        return {
            "success": True,
            "experiment_name": experiment.name,
            "experiment_id": experiment.experiment_id,
            "run": run_data.dict(),
            "message": f"Successfully fetched run with {len(run_data.metrics_history)} metrics"
        }
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Run '{run_id}' not found: {str(e)}"
        )

@app.get("/mlflow/experiments")
async def list_experiments(
    tracking_uri: str = "./mlruns",
    databricks_token: Optional[str] = None
):
    """List all available experiments."""
    try:
        client = get_mlflow_client(tracking_uri, databricks_token)
        experiments = client.search_experiments()
        
        exp_list = [
            {
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "artifact_location": exp.artifact_location,
                "lifecycle_stage": exp.lifecycle_stage
            }
            for exp in experiments
        ]
        
        return {
            "success": True,
            "total_experiments": len(exp_list),
            "experiments": exp_list
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error listing experiments: {str(e)}"
        )

@app.post("/health")
async def health_check(tracking_uri: str = "./mlruns", databricks_token: Optional[str] = None):
    """Health check endpoint."""
    try:
        client = get_mlflow_client(tracking_uri, databricks_token)
        experiments = client.search_experiments(max_results=1)
        mlflow_accessible = True
        experiment_count = len(experiments)
    except Exception as e:
        mlflow_accessible = False
        experiment_count = 0
    
    return {
        "status": "healthy",
        "mlflow_accessible": mlflow_accessible,
        "tracking_uri": tracking_uri,
        "experiments_found": experiment_count
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)