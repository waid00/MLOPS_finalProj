# Employee Insight Engine (GenAI Log Analytics)

An internal MLOps stack that classifies anonymized employee GenAI prompts into actionable categories. The stack combines BERTopic with keyword overrides to flag coding requests, HR policy questions, wellbeing signals, and more, so HR/IT can react to trends rather than individuals.

![Architecture Diagram](assets/architecture.jpeg)

---

## Components

- **Backend API (FastAPI):** [prompt_market/app/backend/main.py](prompt_market/app/backend/main.py) loads a BERTopic artifact and exposes `/health` and `/predict`. Hybrid logic in [prompt_market/app/backend/model.py](prompt_market/app/backend/model.py) applies keyword overrides when confidence is low or critical phrases appear.
- **Frontend (Streamlit):** [prompt_market/app/frontend/streamlit_app.py](prompt_market/app/frontend/streamlit_app.py) offers single-prompt testing, CSV batch upload, and known failure demos. Uses `BACKEND_URL` (defaults to `http://localhost:8000/predict`).
- **ML Pipeline:** Data cleaning in [prompt_market/app/ml/data_processing.py](prompt_market/app/ml/data_processing.py), training in [prompt_market/app/ml/train_model.py](prompt_market/app/ml/train_model.py), topic evaluation in [prompt_market/app/ml/evaluate_topics.py](prompt_market/app/ml/evaluate_topics.py), and hyperparameter sweeps logged to MLflow in [prompt_market/app/ml/mlflow_experiments.py](prompt_market/app/ml/mlflow_experiments.py). Training pulls prompts from Hugging Face `hf://datasets/fka/awesome-chatgpt-prompts/prompts.csv`.
- **Artifacts:** Default model saved to [prompt_market/app/backend/bertopic_model.pkl](prompt_market/app/backend/bertopic_model.pkl); labeled training data cached at [prompt_market/app/backend/labeled_data.csv](prompt_market/app/backend/labeled_data.csv). MLflow runs live under [prompt_market/mlruns](prompt_market/mlruns).

---

## Quick Start (Docker Compose)

From the repo root:

```bash
cd prompt_market
docker-compose up --build
```

Access points:
- Frontend dashboard: http://localhost:8501
- Backend docs: http://localhost:8000/docs

Docker mounts the model file so you can replace `app/backend/bertopic_model.pkl` without rebuilding.

---

## Local Development

### Setup
```bash
# From repo root
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
cd prompt_market
```

### Run services locally
- Backend: `uvicorn app.backend.main:app --reload --host 0.0.0.0 --port 8000`
- Frontend: `BACKEND_URL=http://localhost:8000/predict streamlit run app/frontend/streamlit_app.py`

### Tests
```bash
pytest app/tests/test_app.py
```

---

## Training and Evaluation

- Train a fresh BERTopic model (saves artifact and labeled CSV):
	```bash
	python app/ml/train_model.py \
		--model-path app/backend/bertopic_model.pkl \
		--data-path app/backend/labeled_data.csv
	```

- Evaluate topics and top words:
	```bash
	python app/ml/evaluate_topics.py
	```

- Hyperparameter sweeps with MLflow logging:
	```bash
	python app/ml/mlflow_experiments.py
	mlflow ui  # view results (default http://localhost:5000)
	```

Windows users: the training, evaluation, and test scripts pre-load `c10.dll` to avoid PyTorch load errors; no manual steps needed.

---

## API Reference

- `GET /health` → `{ "status": "healthy", "model_loaded": true }`
- `POST /predict` → send `{ "text": "Write a python script..." }` and receive `{ "topic_id": int, "topic_label": str, "topic_words": [str], "topic_prob": float }`. Empty or short inputs return `Uncategorized / Noise`.

---

## Frontend Workflows

- **Single Prompt:** Enter text and see category, confidence, topic id, and top keywords.
- **Batch CSV:** Upload a CSV with column `prompt`; downloads labeled results as `classified_prompts.csv`.
- **Failure Analysis Demo:** Predefined tricky prompts to observe model limitations and error handling.

---

## Project Structure (key paths)

```plaintext
prompt_market/
├── app/
│   ├── backend/        # FastAPI service + model artifact
│   ├── frontend/       # Streamlit UI
│   ├── ml/             # Data prep, training, evaluation, MLflow runs
│   └── tests/          # API tests
├── docker-compose.yaml # Local orchestration (backend + frontend)
└── mlruns/             # Experiment tracking outputs
```

For business context and modeling decisions, see [business_report.md](business_report.md).
