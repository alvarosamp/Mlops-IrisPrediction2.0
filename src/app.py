import sys
import pickle
import json
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, jsonify
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

app = Flask(__name__, template_folder='templates')

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model_latest.pkl"
model = None
model_metadata = None
IRIS_CLASSES = {0: "setosa", 1: "versicolor", 2: "virginica"}

def load_model():
    global model, model_metadata
    if not MODEL_PATH.exists():
        raise Exception(f"Modelo não encontrado em {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    metadata_path = MODEL_PATH.with_suffix(".metadata.json")
    if metadata_files:
        with open(max(metadata_files, key=lambda p: p.stat().st_mtime), "r") as f:
            model_metadata = json.load(f)
            
try:
    load_model()
except Exception as e:
    print(f"❌ Erro ao carregar o modelo: {e}")
    model = None
    
@app.route("/")
#Pagina inicial
def home():
    model_info = None
    if model_metadata:
        model_info = {
            "accuracy": model_metadata.get("metrics", {}).get("accuracy"),
            "created_at": model_metadata.get("created_at"),
        }
    return render_template("index.html", model_info=model_info)

@app.route("/predict", methods=["POST"])
def predict():
    #Predição via formulário web
    if model is None:
        render_template("index.html", error="Modelo não está carregado.")
        
    try:
        features = [
            float(request.form["sepal_length",0]),
            float(request.form["sepal_width",0]),
            float(request.form["petal_length",0]),
            float(request.form["petal_width",0]),
        ]
        prediction_num = model.predict([features])[0]
        prediction_class = IRIS_CLASSES.get(prediction_num, "Desconhecido")
        return render_template("index.html", prediction=prediction_class)
    except Exception as e:
        return render_template("index.html", error=f"Erro na predição: {e}")
    
@app.route("/api/predict", methods=["POST"])
def api_predict():
    #Predição via API
    if model is None:
        return jsonify({"error": "Modelo não está carregado."}), 500
    try:
        data = request.get_json()
        features = data.get('features', [])
        if len(features) != 4:
            return jsonify({"error": "Quatro características são necessárias."}), 400
        features_array = np.array(features).reshape(1, -1)
        prediction_num = int(model.predict(features_array)[0])
        response = {
            "prediction" ; prediction_num,
            "prediction_class" : IRIS_CLASSES.get(prediction_num, "Desconhecido")
            "probabilities" : model.predict_proba(features_array).tolist() if hasattr(model, "predict_proba") else None,
            "timestamp" : datetime.now().isoformat()
        }
        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": f"Erro na predição: {e}"}), 500

@app.route("/health", methods=["GET"])
def health():
    if model is None:
        return jsonify({"status": "unhealthy", "message": "Modelo não está carregado."}), 500
    return jsonify({"status": "healthy", "message": "Modelo está carregado."}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug = True)