import os
import json
import yaml
import hashlib
from datetime import datetime
from pathlib import Path
try:
    import mlflow  # type: ignore
    HAS_MLFLOW = True
except Exception:
    mlflow = None  # type: ignore
    HAS_MLFLOW = False
import subprocess


def get_model_hash(model_path):
    hash_md5 = hashlib.md5()
    with open(model_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def save_model_metadata(model_path, metrics, params):
    metadata = {
        "model_path": str(model_path),
        "model_hash": get_model_hash(model_path),
        "created_at": datetime.now().isoformat(),
        "metrics": metrics,
        "params": params
    }

    output_path = Path(model_path).with_suffix(".metadata.json")
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"🔖 Metadados salvos em: {output_path}")
    return output_path


def load_config(path="config/model_config.yaml"):
    config_path = Path(path)
    if not config_path.is_absolute():
        project_root = Path(__file__).resolve().parent.parent
        config_path = project_root / config_path
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def dvc_track(model_path):
    """
    Executa DVC no HOST, nunca no container.
    """
    try:
        subprocess.run(["dvc", "add", model_path], check=True)
        print(f"📌 DVC adicionou: {model_path}")
        return True
    except Exception as e:
        print(f"❌ Erro ao adicionar modelo ao DVC: {e}")
        return False


def setup_mlflow(tracking_uri, experiment_name):
    if not HAS_MLFLOW:
        print("ℹ️ MLflow não instalado; prosseguindo sem tracking.")
        return
    try:
        # Permite sobrepor via variável de ambiente (útil em Docker)
        env_uri = os.getenv("MLFLOW_TRACKING_URI")
        effective_uri = env_uri if env_uri else tracking_uri
        mlflow.set_tracking_uri(effective_uri)
        mlflow.set_experiment(experiment_name)
        print(f"📡 MLflow configurado: {effective_uri} → {experiment_name}")
    except Exception as e:
        # Fallback to local file store if server unavailable
        local_store = Path(__file__).resolve().parent.parent / "mlruns"
        mlflow.set_tracking_uri(f"file://{local_store}")
        mlflow.set_experiment(experiment_name)
        print(f"📡 MLflow local (fallback) em {local_store}: {e}")
