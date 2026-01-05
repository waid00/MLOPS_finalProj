PromptMarket: End-to-End Topic Modeling

Project Description

PromptMarket is a Machine Learning application designed to categorize ChatGPT prompts into business-relevant topics (e.g., Coding, Marketing, Roleplay). It uses BERTopic for clustering, FastAPI for inference, and Streamlit for user interaction.

Setup & Run

Prerequisites

Docker & Docker Compose

Python 3.9+ (for local training)

1. Train the Model (Local)

Before running Docker, you must train the model to generate the artifact.

# Install dependencies
pip install -r app/backend/requirements.txt

# Run training script (Saved to app/backend/bertopic_model.pkl)
python app/ml/train_model.py


2. Run the App (Docker)

docker-compose up --build


Frontend: http://localhost:8501

Backend Docs: http://localhost:8000/docs

3. MLflow Experiments

To view experiments:

mlflow ui
# Open http://localhost:5000


Sample Inputs

Input: "Act as a Linux terminal. I want you to reply only with the terminal output inside one unique code block."

Expected Output: Topic: Coding & Development, Key words: linux, terminal, command, bash

Team

[Your Name] - ML Architect