import pickle
from pathlib import Path
from datetime import datetime

import mlflow
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from inspect import signature
from sklearn.metrics import accuracy_score

from utils import load_config, setup_mlflow, save_model_metadata, dvc_track


def train(config_path="config/model_config.yaml"):

    config = load_config(config_path)
    setup_mlflow(
        config["mlflow"]["tracking_uri"],
        config["mlflow"]["experiment_name"]
    )

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