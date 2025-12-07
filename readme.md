## Mlops-IrisPrediction 2.0

Um projeto completo de MLOps para classificação do dataset Iris usando Random Forest, com:

- Treinamento reproducível (scikit-learn) e logging em MLflow
- API Flask com página web e endpoint REST para predições
- Artefatos versionados: aliases estáveis e metadados (hash, métricas, hiperparâmetros)
- Integração com DVC para versionar modelos no repositório (no host)
- Orquestração via Docker Compose

---

## Visão geral

- Modelo: `RandomForestClassifier` treinado no Iris
- API: Flask servindo uma página simples e um endpoint JSON
- MLflow: UI disponível via Docker; experiment name configurável
- Modelos salvos em `models/` com arquivos timestamp e aliases estáveis:
	- `model_YYYYMMDD_HHMMSS.pkl`
	- `model_latest.pkl` e `iris_model_latest.pkl`
- Metadados por modelo: `model_*.metadata.json` com hash, métricas, params, data

Estrutura principal:

```
src/
	app.py        # API Flask
	train.py      # Treinamento do modelo
	utils.py      # Utilitários: config, mlflow, dvc, metadata
templates/
	index.html    # Página web com formulário e infos do modelo
models/         # Artefatos de modelo (.pkl) + metadados (.metadata.json)
config/
	model_config.yaml  # Hiperparâmetros, dados e MLflow
docker/
	Dockerfile.api, Dockerfile.train, Dockerfile.mlflow
docker-compose.yaml
```

---

## Requisitos

- Docker e Docker Compose
- Opcional (desenvolvimento local): Python 3.10+ e `pip`

---

## Rodando com Docker

1) Suba os serviços (API, MLflow, job de treino sob demanda):

```bash
docker compose up -d
```

Portas padrão:
- API: http://localhost:5001 (pode alterar com `API_PORT`)
- MLflow UI: http://localhost:5003 (pode alterar com `MLFLOW_PORT`)

Exemplo com portas customizadas:

```bash
API_PORT=8080 MLFLOW_PORT=5500 docker compose up -d
```

2) Treine o modelo (gera `model_latest.pkl` e metadados automaticamente):

```bash
docker compose run --rm train
```

Os artefatos ficam em `models/` (montado como volume no host). Após o treino, a API consegue carregar o modelo automaticamente.

3) Verifique a saúde da API:

```bash
curl -s http://localhost:5001/health
```

---

## Uso da API

- Página web: http://localhost:5001
	- Preencha o formulário; o resultado aparece como “Previsão: <classe>”.

- Endpoint REST:

```bash
curl -s -X POST -H 'Content-Type: application/json' \
	http://localhost:5001/api/predict \
	-d '{"features":[5.1,3.5,1.4,0.2]}'
```

Resposta (exemplo):

```json
{
	"prediction": 0,
	"prediction_class": "setosa",
	"probabilities": [[1.0, 0.0, 0.0]],
	"timestamp": "2025-12-07T20:24:56.059685"
}
```

- Metadados completos do modelo carregado:

```bash
curl -s http://localhost:5001/model/info | jq .
```

---

## Treinamento

Parâmetros e dados são definidos em `config/model_config.yaml`:

- Hiperparâmetros (RandomForest): `model.*`
- Split de dados: `data.*`
- MLflow: `mlflow.experiment_name` e `mlflow.tracking_uri`

Ao final do treino:
- Salva `models/model_YYYYMMDD_HHMMSS.pkl`
- Cria aliases estáveis: `models/model_latest.pkl` e `models/iris_model_latest.pkl`
- Salva metadados: `models/model_*.metadata.json` com:
	- `model_hash`, `created_at`, `metrics.accuracy`, `params`, `model_path`

### MLflow

- UI: http://localhost:5003
- Dentro do Docker, o tracking é apontado automaticamente para `http://mlflow:5000`.
- No host, você pode usar o valor padrão (`http://localhost:5000`) do arquivo de config ou definir `MLFLOW_TRACKING_URI` manualmente.

### DVC

O DVC deve ser executado no host (fora do container). As tentativas de `dvc add` dentro do container são ignoradas se o repo não existir lá.

Para versionar o último modelo no host:

```bash
dvc add models/model_YYYYMMDD_HHMMSS.pkl
git add models/*.dvc .gitignore
git commit -m "Track model artifact with DVC"
```

Se quiser automatizar, podemos adicionar um script/Makefile que rode `dvc add` localmente após o treino.

---

## Desenvolvimento local (sem Docker)

1) Crie e ative um virtualenv (recomendado Python 3.10 para compatibilidade com Dockerfiles):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Treine localmente:

```bash
python src/train.py
```

3) Rode a API local:

```bash
python src/app.py  # http://localhost:5001
```

4) Testes:

```bash
pytest -q
```

Obs.: os testes procuram por um modelo existente; se não houver, o teste dispara um treino automaticamente.

---

## Troubleshooting

- Porta 5000 ocupada
	- O `docker-compose.yaml` já parametriza as portas. Use `MLFLOW_PORT` para trocar a porta exposta do MLflow (padrão 5003) e `API_PORT` para a API (padrão 5001).

- API sobe mas não encontra o modelo
	- Rode `docker compose run --rm train` para gerar `model_latest.pkl` em `models/`.

- TemplateNotFound: `index.html`
	- Já configurado para apontar para `templates/` no caminho absoluto dentro do container. Se customizar os caminhos, garanta que `templates/` é copiado em `Dockerfile.api`.

- DVC erro no container
	- É esperado: o repositório Git/DVC não existe dentro do container. Execute `dvc add` no host.

---

## Licença

Este projeto é fornecido como está, para fins educacionais e de demonstração de práticas de MLOps.

