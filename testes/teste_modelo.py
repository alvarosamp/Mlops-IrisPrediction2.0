import sys
import pytest
import pickle
from pathlib import Path

# Resolve o diretório raiz do projeto a partir deste arquivo de teste
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

def _locate_or_train_model():
    """
    Retorna o caminho de um modelo disponível:
    - Tenta usar iris_model_latest.pkl ou model_latest.pkl
    - Caso não exista, tenta encontrar o .pkl mais recente em models/
    - Se ainda não existir, roda o treino para gerar um modelo
    """
    candidates = [
        MODELS_DIR / "iris_model_latest.pkl",
        MODELS_DIR / "model_latest.pkl",
    ]
    for c in candidates:
        if c.exists():
            return c

    # Busca o .pkl mais recente (ignora arquivos .dvc)
    if MODELS_DIR.exists():
        pkls = sorted(
            [p for p in MODELS_DIR.glob("*.pkl") if not p.name.endswith(".dvc")],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if pkls:
            return pkls[0]

    # Não há modelo: executa treino rapidamente
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.train import train  # type: ignore
    generated_path = train()

    # Após treino, prefere alias estáveis se existirem
    for c in candidates:
        if c.exists():
            return c
    return Path(generated_path)


def test_model_exists():
    model_path = _locate_or_train_model()
    assert model_path.exists(), "Modelo não encontrado e não foi possível treinar automaticamente."


def test_model_prediction():
    model_path = _locate_or_train_model()

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    prediction = model.predict([[5.1, 3.5, 1.4, 0.2]])
    assert prediction is not None
    assert 0 <= prediction[0] <= 2