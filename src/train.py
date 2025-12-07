import pickle
import shutil
from pathlib import Path
from datetime import datetime

try:
    import mlflow  # type: ignore
except Exception:
    mlflow = None  # type: ignore
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from inspect import signature
from sklearn.metrics import accuracy_score

import os
from utils import load_config, setup_mlflow, save_model_metadata, dvc_track, HAS_MLFLOW


def train(config_path="config/model_config.yaml"):

    config = load_config(config_path)
    if HAS_MLFLOW:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"])
        setup_mlflow(tracking_uri, config["mlflow"]["experiment_name"])

    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"]
    )

    raw_params = config["model"]
    rf_sig = signature(RandomForestClassifier.__init__)
    allowed = set(rf_sig.parameters.keys())
    allowed.discard('self')
    params = {k: v for k, v in raw_params.items() if k in allowed}
    model = RandomForestClassifier(**params)

    if mlflow:
        with mlflow.start_run():
            mlflow.log_params(params)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            mlflow.log_metric("accuracy", accuracy)

            print(f"🔍 Acurácia: {accuracy:.4f}")

            # Ensure models directory resolves relative to project root
            project_root = Path(__file__).resolve().parent.parent
            models_dir = project_root / "models"
            models_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = models_dir / f"model_{timestamp}.pkl"

            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            mlflow.log_artifact(str(model_path))

            # Criar aliases estáveis para facilitar consumo por app/testes
            try:
                latest_path = models_dir / "model_latest.pkl"
                iris_latest_path = models_dir / "iris_model_latest.pkl"
                shutil.copyfile(model_path, latest_path)
                shutil.copyfile(model_path, iris_latest_path)
                print(f"🔗 Aliases atualizados: {latest_path.name}, {iris_latest_path.name}")
            except Exception as e:
                print(f"⚠️ Não foi possível atualizar aliases de modelo: {e}")

            save_model_metadata(model_path, {"accuracy": accuracy}, params)

            use_dvc = (
                config.get("versioning", {}).get("use_dvc") or
                config.get("versioning", {}).get("auto_dvc")
            )
            if use_dvc:
                dvc_track(str(model_path))

            print(f"✅ Modelo salvo em {model_path}")
            return model_path
    else:
        # Fluxo sem MLflow
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"🔍 Acurácia: {accuracy:.4f}")

        project_root = Path(__file__).resolve().parent.parent
        models_dir = project_root / "models"
        models_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = models_dir / f"model_{timestamp}.pkl"

        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Aliases estáveis
        try:
            latest_path = models_dir / "model_latest.pkl"
            iris_latest_path = models_dir / "iris_model_latest.pkl"
            shutil.copyfile(model_path, latest_path)
            shutil.copyfile(model_path, iris_latest_path)
            print(f"🔗 Aliases atualizados: {latest_path.name}, {iris_latest_path.name}")
        except Exception as e:
            print(f"⚠️ Não foi possível atualizar aliases de modelo: {e}")

        save_model_metadata(model_path, {"accuracy": accuracy}, params)

        use_dvc = (
            config.get("versioning", {}).get("use_dvc") or
            config.get("versioning", {}).get("auto_dvc")
        )
        if use_dvc:
            dvc_track(str(model_path))

        print(f"✅ Modelo salvo em {model_path}")
        return model_path


if __name__ == "__main__":
    train()