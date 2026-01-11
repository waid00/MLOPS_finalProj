# Employee Insight Engine (GenAI Log Analytics)

An internal MLOps stack that classifies anonymized employee GenAI prompts into actionable categories. The stack combines BERTopic with keyword overrides to flag coding requests, HR policy questions, wellbeing signals, and more, enabling HR/IT teams to react to trends rather than individuals. 

![Architecture Diagram](assets/architecture.jpeg)

---

## Overview

This project implements a production-ready topic classification system using: 
- **BERTopic** for unsupervised topic modeling
- **Sentence Transformers** (`all-mpnet-base-v2`) for semantic embeddings
- **Hybrid classification** combining ML predictions with keyword-based overrides
- **MLflow** for experiment tracking and model versioning
- **Docker Compose** for local development and deployment

**Tech Stack:** Python 3.9+, FastAPI, Streamlit, BERTopic, PyTorch (CPU), MLflow, Docker

---

## Components

### Backend API (FastAPI)
- **Location:** [`prompt_market/app/backend/`](prompt_market/app/backend/)
- **Main file:** [`main.py`](prompt_market/app/backend/main.py)
- **Model logic:** [`model.py`](prompt_market/app/backend/model.py)
- **Features:**
  - REST API with `/health` and `/predict` endpoints
  - Hybrid classification with keyword override rules
  - Automatic model loading on startup with lazy-load fallback
  - Production-ready logging and error handling

### Frontend (Streamlit)
- **Location:** [`prompt_market/app/frontend/`](prompt_market/app/frontend/)
- **Main file:** [`streamlit_app.py`](prompt_market/app/frontend/streamlit_app. py)
- **Features:**
  - Single-prompt testing interface
  - Batch CSV upload and download
  - Failure analysis demo with edge cases
  - Real-time classification with confidence scores

### ML Pipeline
- **Location:** [`prompt_market/app/ml/`](prompt_market/app/ml/)
- **Key files:**
  - [`data_processing.py`](prompt_market/app/ml/data_processing.py) - Data cleaning and preprocessing
  - [`train_model.py`](prompt_market/app/ml/train_model.py) - BERTopic model training
  - [`evaluate_topics.py`](prompt_market/app/ml/evaluate_topics.py) - Topic evaluation and analysis
  - [`mlflow_experiments.py`](prompt_market/app/ml/mlflow_experiments.py) - Hyperparameter optimization
- **Data source:** [Hugging Face dataset](https://huggingface.co/datasets/fka/awesome-chatgpt-prompts)

### Artifacts
- **Model:** `prompt_market/app/backend/bertopic_model.pkl/` (directory-based BERTopic artifact)
- **Training data:** `prompt_market/app/backend/labeled_data.csv`
- **MLflow runs:** `mlruns/` (at repository root)
- **Processed data:** `processed_prompts.csv` (at repository root)

---

## Quick Start (Docker Compose)

### Prerequisites
- Docker and Docker Compose installed
- 2GB+ free RAM (for embedding model)

### Run the Application

From the repository root:

```bash
cd prompt_market
docker-compose up --build
```

**Access points:**
- Frontend UI: [http://localhost:8501](http://localhost:8501)
- Backend API: [http://localhost:8000](http://localhost:8000)
- API Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)

The Docker setup automatically mounts the model file, so you can update `app/backend/bertopic_model.pkl` without rebuilding containers.

---

## Local Development

### Setup Environment

```bash
# From repository root
python -m venv .venv

# Activate virtual environment
. venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Run Services Locally

**Terminal 1 - Backend:**
```bash
cd prompt_market
uvicorn app.backend.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd prompt_market
set BACKEND_URL=http://localhost:8000/predict     # Windows
# export BACKEND_URL=http://localhost:8000/predict  # macOS/Linux
streamlit run app/frontend/streamlit_app.py
```

### Run Tests

```bash
cd prompt_market
pytest app/tests/test_app.py -v
```

**Test coverage includes:**
- Health check endpoint validation
- Keyword override verification (coding, creative writing)
- Edge case handling (empty input, invalid JSON)
- Model loading and lifecycle

---

## Training and Evaluation

### Train a New Model

```bash
cd prompt_market
python app/ml/train_model.py \
    --model-path app/backend/bertopic_model.pkl \
    --data-path app/backend/labeled_data.csv
```

**Training configuration:**
- Embedding model: `all-mpnet-base-v2`
- UMAP:  15 neighbors, 5 components
- HDBSCAN: min_cluster_size=10, min_samples=5
- Custom stopwords for domain-specific filtering

### Evaluate Topics

View topic distributions and top keywords:

```bash
cd prompt_market
python app/ml/evaluate_topics.py
```

### Hyperparameter Optimization

Run systematic experiments with MLflow tracking:

```bash
cd prompt_market
python app/ml/mlflow_experiments.py
```

Then view results: 
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Access the MLflow UI at [http://localhost:5000](http://localhost:5000)

**Note for Windows users:** Training scripts automatically handle `c10.dll` loading issues with PyTorch. No manual intervention needed. 

---

## API Reference

### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Predict Topic
```http
POST /predict
Content-Type: application/json

{
  "text": "Write a Python script to calculate prime numbers"
}
```

**Response:**
```json
{
  "topic_id": 0,
  "topic_label":  "Coding & Development",
  "topic_words": ["code", "python", "script", "function", "algorithm"],
  "topic_prob": 0.98
}
```

**Special cases:**
- Empty or very short inputs → `"Uncategorized / Noise"`
- Keyword matches override ML predictions for high-confidence categories

---

## Frontend Features

### 1. Single Prompt Mode
- Real-time classification
- Displays category, confidence score, topic ID, and top keywords
- Useful for testing specific inputs

### 2. Batch CSV Upload
- Upload CSV with `prompt` column
- Processes all rows and returns labeled results
- Download as `classified_prompts.csv`
- Progress bar for large batches

### 3. Failure Analysis Demo
- Predefined edge cases demonstrating model limitations
- Helps understand boundary conditions
- Includes short inputs, ambiguous text, and multilingual examples

---

## Project Structure

```plaintext
MLOPS_finalProj/
├── README.md                    # This file
├── business_report.md           # Business context and modeling decisions
├── requirements.txt             # Python dependencies (root level)
├── mlruns/                      # MLflow experiment tracking
├── mlflow. db                    # SQLite backend for MLflow
├── processed_prompts.csv        # Cached preprocessed training data
├── assets/
│   └── architecture.jpeg        # System architecture diagram
└── prompt_market/
    ├── docker-compose.yaml      # Container orchestration
    └── app/
        ├── backend/             # FastAPI service
        │   ├── main.py          # API endpoints
        │   ├── model.py         # Classifier with hybrid logic
        │   ├── Dockerfile       # Backend container config
        │   ├── requirements.txt # Backend dependencies
        │   ├── bertopic_model.pkl/  # Trained model artifact
        │   └── labeled_data.csv     # Training data with labels
        ├── frontend/            # Streamlit UI
        │   ├── streamlit_app.py # UI application
        │   ├── Dockerfile       # Frontend container config
        │   └── requirements.txt # Frontend dependencies
        ├── ml/                  # Training pipeline
        │   ├── data_processing.py    # ETL and cleaning
        │   ├── train_model.py        # Model training script
        │   ├── evaluate_topics.py    # Evaluation utilities
        │   └── mlflow_experiments.py # Hyperparameter tuning
        └── tests/               # Integration tests
            └── test_app. py      # API endpoint tests
```

---

## Key Features

### Hybrid Classification
Combines BERTopic's unsupervised learning with domain-specific keyword rules: 
- **Technical & Coding** - Detects programming languages, technical terms
- **HR & Employment** - Identifies policy questions, benefits inquiries
- **Employee Wellbeing** - Flags burnout, stress, mental health signals
- **IT Support** - Recognizes infrastructure and access issues
- **Legal & Compliance** - Catches NDA, GDPR, policy mentions
- **Finance & Budgeting** - Identifies invoice, budget, expense topics
- **Learning & Development** - Detects training and tutorial requests

### Production-Ready Features
- Proper logging instead of print statements
- Pydantic validation for API inputs
- Health check endpoint for monitoring
- Lazy-load fallback for model failures
- Docker multi-stage builds with CPU-only PyTorch
- Environment-based configuration
- Comprehensive test suite

### MLOps Best Practices
- Experiment tracking with MLflow
- Reproducible training pipelines
- Version-controlled model artifacts
- Automated hyperparameter optimization
- Model evaluation metrics

---

## Platform Notes

### Windows Compatibility
All Python scripts include automatic fixes for the common PyTorch `c10.dll` loading error. The following files have built-in patches:
- `train_model.py`
- `evaluate_topics.py`
- `mlflow_experiments.py`
- `test_app.py`

### Docker Optimization
Backend Dockerfile installs CPU-only PyTorch before other dependencies to avoid downloading large CUDA packages (~4GB savings).

---

## Troubleshooting

### Backend doesn't load model
- Check that `prompt_market/app/backend/bertopic_model.pkl/` exists
- Run training script: `python app/ml/train_model.py`
- Check logs for DLL errors (Windows) or memory issues

### Frontend can't connect to backend
- Verify backend is running on port 8000
- Check `BACKEND_URL` environment variable
- For Docker:  use `http://backend:8000/predict`
- For local: use `http://localhost:8000/predict`

### MLflow UI not showing experiments
- Ensure you're in the repository root
- Use:  `mlflow ui --backend-store-uri sqlite:///mlflow.db`
- Check that `mlflow.db` exists

---

For detailed business context, modeling decisions, and performance analysis, see [business_report.md](business_report.md).

---