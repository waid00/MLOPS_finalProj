# Employee Insight Engine (GenAI Log Analytics)

An internal MLOps solution designed to analyze anonymized employee interactions with Generative AI tools. By applying unsupervised topic modeling (BERTopic), this tool uncovers workforce trends, detects stress signals (e.g., "burnout", "health"), and identifies skill gaps to help HR and Management provide targeted support.

---

## Architecture

The application follows a microservices architecture using Docker containers:

- **Frontend (Streamlit):** An interactive dashboard for HR managers to visualize topic trends and test manual inputs.

- **Backend (FastAPI):** A REST API that hosts the trained BERTopic model and handles inference requests.

- **ML Pipeline:** Offline training scripts that process data, train the model, and track experiments via MLflow.

![Architecture Diagram](assets/architecture.jpeg)

---

## Quick Start (Docker)

The easiest way to run the application is using Docker Compose. This handles all dependencies automatically.

### Navigate to the project directory:

```bash
cd prompt_market
```

### Build and Run:

```bash
docker-compose up --build
```

### Access the App:

- **Frontend (HR Dashboard):** http://localhost:8501

- **Backend API Docs:** http://localhost:8000/docs

- **MLflow UI:** http://localhost:5000 (if running experiments)

---

## Local Development & Training

If you need to retrain the model or run tests locally without Docker.

### 1. Prerequisites

- Python 3.10+

- Virtual Environment (Recommended)

### 2. Installation

```bash
cd prompt_market
# Create virtual env
python -m venv .venv
# Activate (Windows)
.venv\Scripts\activate
# Activate (Mac/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Training the Model

To generate a new bertopic_model.pkl artifact based on the latest data:

```bash
python app/ml/train_model.py
```

**Note for Windows Users:** The script includes a specialized patch to handle PyTorch c10.dll loading issues.

### 4. Running MLflow Experiments

To explore hyperparameters (n_neighbors, min_cluster_size) and visualize metrics:

```bash
python app/ml/mlflow_experiments.py
mlflow ui
```

### 5. Evaluating the Model

To verify the generated topics and outlier ratios:

```bash
python app/ml/evaluate_topics.py
```

### 6. Running Tests

To validate the hybrid rule-based system and API endpoints:

```bash
pytest app/tests/test_app.py
```

---

## Key Features

- **Hybrid Classification:** Combines state-of-the-art BERTopic (semantic clustering) with deterministic rules (keyword overrides) for critical categories like Coding or Creative Writing.

- **Privacy-First:** Designed to analyze aggregated trends rather than individual surveillance.

- **Robustness:** Handles edge cases (empty inputs, short prompts) via extensive error handling and Uncategorized labeling.

- **Windows Compatibility:** Includes custom patches for PyTorch DLL initialization on Windows environments.

---

## Project Structure

```plaintext
prompt_market/
├── app/
│   ├── backend/          # FastAPI application
│   │   ├── main.py       # API Endpoints
│   │   ├── model.py      # Inference Logic (Hybrid System)
│   │   └── bertopic_model.pkl # Trained Artifact
│   ├── frontend/         # Streamlit Dashboard
│   │   └── streamlit_app.py
│   ├── ml/               # Machine Learning Pipeline
│   │   ├── train_model.py       # Training Script
│   │   ├── mlflow_experiments.py # Hyperparameter Tuning
│   │   └── evaluate_topics.py   # Quality Check
│   └── tests/            # Pytest Suite
├── docker-compose.yaml   # Container Orchestration
└── requirements.txt      # Python Dependencies
```
