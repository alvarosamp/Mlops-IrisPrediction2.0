"""Funçoes utilitárias para o mlops"""
import os
import json 
import yaml
import hashlib
from datetime import datetime
from pathlib import Path
import mlflow

def get_model_hash(model_path):
    """Gera um hash MD5 para o arquivo do modelo especificado."""
    hash_md5 = hashlib.md5()
    with open(model_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def save_model_metadata(model_path, metrics, params , model_version= None):
    """Salva metadados do modelo"""
    metadata = {
        "model_path": str(model_path),
        "model_hash": get_model_hash(model_path),
        "created_at": datetime.now().isoformat(),
        "metrics": metrics,
        "params": params,
        "model_version": model_version
    }
    
    metadata_path = Path(model_path).parent / f"{Path(model_path).stem}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    
    print(f"✅ Metadados salvos em: {metadata_path}")
    return metadata_path

def version_model_with_dvc(model_path):
    """Versiona modelo com DVC"""
    import subprocess
    try:
        result = subprocess.run(["dvc", "add", model_path], capture_output=True, text=True, check=True)
        print(f"✅ Modelo versionado com DVC: {result.stdout}")
        dvc_file = f"{model_path}.dvc"
        subprocess.run(["git", "add", dvc_file], check=True)
        subprocess.run(["git", "add", ".gitignore"], check=True)
        print(f"✅ Arquivos DVC adicionados ao Git: {dvc_file} e .gitignore")
        return True 
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro ao versionar modelo com DVC: {e.stderr}")
        return False
    except FileNotFoundError:
        print("❌ DVC não está instalado ou não foi encontrado no PATH.")
        return False
    

def setup_mlflow(tracking_uri = None, experiment_name = None):
    """Configura MLflow com URI de rastreamento e nome do experimento"""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        print(f"✅ MLflow tracking URI configurado para: {tracking_uri}")
    if experiment_name:
        mlflow.set_experiment(experiment_name)
        print(f"✅ Experimento MLflow definido para: {experiment_name}")
        