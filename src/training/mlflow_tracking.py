"""
MLflow Experiment Tracking for ECG Arrhythmia Detection
========================================================
Logs experiments, parameters, metrics, and models to MLflow.
"""

import logging
from pathlib import Path
from contextlib import contextmanager

import mlflow
import mlflow.pytorch
import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_mlflow_config() -> dict:
    """Load MLflow configuration from config.yaml."""
    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    if config_path.exists():
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        return cfg.get("mlflow", {})
    return {}


@contextmanager
def mlflow_run(experiment_name: str = None, run_name: str = None, config: dict = None):
    """
    Context manager for MLflow runs.

    Usage:
        with mlflow_run("my_experiment", "run_1") as run:
            mlflow.log_param("lr", 0.001)
            mlflow.log_metric("accuracy", 0.95)

    Parameters
    ----------
    experiment_name : str, optional
        MLflow experiment name.
    run_name : str, optional
        Name for this run.
    config : dict, optional
        Configuration to log.

    Yields
    ------
    mlflow.Run
        The active MLflow run.
    """
    mlflow_config = load_mlflow_config()

    if experiment_name is None:
        experiment_name = mlflow_config.get("experiment_name", "ecg-arrhythmia-detection")

    tracking_uri = mlflow_config.get("tracking_uri", str(PROJECT_ROOT / "mlruns"))
    mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        # Log config if provided
        if config:
            _log_config(config)

        logger.info(f"MLflow run started: {run.info.run_id}")
        logger.info(f"Experiment: {experiment_name}")
        logger.info(f"Tracking URI: {tracking_uri}")

        try:
            yield run
        finally:
            logger.info(f"MLflow run finished: {run.info.run_id}")


def _log_config(config: dict, prefix: str = ""):
    """Log configuration parameters to MLflow."""
    for key, value in config.items():
        param_name = f"{prefix}{key}" if prefix else key
        if isinstance(value, dict):
            _log_config(value, prefix=f"{param_name}.")
        elif isinstance(value, (int, float, str, bool)):
            mlflow.log_param(param_name, value)
        else:
            mlflow.log_param(param_name, str(value))


def log_training_metrics(epoch: int, metrics: dict):
    """
    Log training metrics for an epoch.

    Parameters
    ----------
    epoch : int
        Current epoch number.
    metrics : dict
        Metrics to log (train_loss, val_loss, train_acc, val_acc, learning_rate).
    """
    for key, value in metrics.items():
        mlflow.log_metric(key, value, step=epoch)


def log_evaluation_results(results: dict, prefix: str = ""):
    """
    Log evaluation results.

    Parameters
    ----------
    results : dict
        Evaluation results.
    prefix : str
        Prefix for metric names.
    """
    for key, value in results.items():
        if isinstance(value, (int, float)):
            metric_name = f"{prefix}{key}" if prefix else key
            mlflow.log_metric(metric_name, value)


def log_model(model, model_name: str = "ecg_model"):
    """
    Log a PyTorch model to MLflow.

    Parameters
    ----------
    model : nn.Module
        The model to log.
    model_name : str
        Name for the model artifact.
    """
    mlflow.pytorch.log_model(model, model_name)
    logger.info(f"Model logged to MLflow as '{model_name}'")


def log_confusion_matrix(cm, class_names: list[str]):
    """
    Log confusion matrix as an artifact.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix.
    class_names : list of str
        Class names.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    # Save to temp file and log
    temp_path = Path(PROJECT_ROOT / "mlruns" / "confusion_matrix.png")
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(temp_path, dpi=150, bbox_inches="tight")
    plt.close()

    mlflow.log_artifact(str(temp_path))
    temp_path.unlink(missing_ok=True)


def log_plot(plot_path: str | Path, artifact_name: str):
    """
    Log a plot file as an MLflow artifact.

    Parameters
    ----------
    plot_path : str or Path
        Path to the plot file.
    artifact_name : str
        Name for the artifact.
    """
    mlflow.log_artifact(str(plot_path), artifact_name=artifact_name)
