# End-to-End ML Topic Modeling: Employee Insight Engine

## Executive Summary

This project implements an internal MLOps solution designed to analyze anonymized employee interactions with Generative AI tools. By applying advanced topic modeling to prompt logs, we aim to uncover hidden workforce trends, identify stress signals (e.g., health-related searches), and detect skill gaps. This proactive insight allows HR and management to offer targeted support, such as wellness programs or training, improving overall employee retention and satisfaction.

---

## A. ML Problem Framing

**Business Goal:** Monitor and categorize internal Generative AI usage to identify actionable insights regarding employee well-being, workload challenges, and professional needs.

**ML Problem:** Unsupervised Semantic Clustering (Topic Modeling) combined with a Rule-Based Hybrid System for specific flag detection.

**Metrics:**
- **Business:** Actionable Insight Rate (percentage of topics leading to HR intervention) and Employee Satisfaction Score (correlated trends).
- **ML:** Topic Coherence, Outlier Ratio, and Cluster Purity.

**Stakeholders:** HR Department (Wellness officers), Team Leads, and Internal Operations.

---

## B. Data Processing Decisions

- **Anonymization (Conceptual):** In a real deployment, PII (Personally Identifiable Information) would be stripped. For this prototype, we treat the dataset as pre-anonymized logs.
- **Deduplication:** Removed exact text matches to prevent frequency bias from automated scripts.
- **Length Filtering:** Removed prompts < 4 words to filter out casual chat noise ("Hi", "Thanks").
- **Preprocessing:** Lowercasing and whitespace normalization. We maintained full sentence structures to allow the model to capture the intent and emotion behind the prompt, not just keywords.
- **Stopword Removal:** Added custom stopwords (e.g., "chatgpt", "write") to focus on the subject of the request (e.g., "burnout", "python error", "deadline").

---

## C. Model Development

We utilized **BERTopic** with the following final configuration:

- **Embedding Model:** all-mpnet-base-v2 (High semantic accuracy to detect subtle nuances in intent).
- **Dimensionality Reduction:** UMAP (n_neighbors=15, n_components=5).
- **Clustering:** HDBSCAN (min_cluster_size=7, min_samples=2).
- **Results:** Identified distinct categories including "Technical Troubleshooting", "Creative Writing", and "Health & Wellness" (simulated).

### Hyperparameter Justification

We systematically explored the parameter space to ensure we capture meaningful employee intents rather than just keywords.

#### 1. Decision: Embedding Model (all-mpnet-base-v2 vs all-MiniLM-L6-v2)

**Experiment:** We started with all-MiniLM-L6-v2 for speed.

**Observation:** It struggled to differentiate between "debug code" (work task) and "learning to code" (training need).

**Decision:** We switched to all-mpnet-base-v2. Its deeper semantic understanding is critical for correctly categorizing sensitive topics like stress or health queries versus general work tasks.

#### 2. Decision: UMAP Neighbors (n_neighbors: 5 vs 15 vs 30)

**Experiment:** We tested local (5) vs global (30) manifold approximations.

**Observation:** n_neighbors=5 fragmented "Health" topics into too many specific ailments, making HR reporting difficult. n_neighbors=30 merged them too broadly.

**Decision:** n_neighbors=15 provided the optimal granularity for department-level reporting.

#### 3. Decision: HDBSCAN Cluster Size (min_cluster_size: 5 vs 7)

**Decision:** We increased the minimum size to 7 to avoid reacting to one-off random searches. We want to identify trends (groups of employees having similar issues), not spy on individuals.

### Experimentation Evidence

Below are screenshots from our MLflow dashboard demonstrating the iterative process:

> [INSERT SCREENSHOT 1 HERE: MLflow Table]

**Figure 1:** Comparison of model runs to optimize topic granularity.

> [INSERT SCREENSHOT 2 HERE: MLflow Artifacts]

**Figure 2:** Validation of model artifacts saved during the training pipeline.

---

## D. Failure Analysis & Limitations

### Failure Cases Table

| Input Prompt | Prediction | Root Cause | HR Impact & Mitigation |
|-------------|-----------|-----------|----------------------|
| "Fix this." | Coding (Low Confidence) | Lack of Context. "Fix" implies broken workflows or code, but without more info, we can't determine the domain. | Ignore Low Confidence. HR should not be alerted on vague frustrations. Only flags with >80% confidence trigger reports. |
| "I feel tired." | Health / Wellness | Success Case. The model correctly identifies sentiment/state. | Aggregate Reporting. To protect privacy, this contributes to a "Team Burnout Index" rather than flagging the individual. |
| "Code a poem about flowers." | Coding | Mixed Intent. The user is likely taking a break, but the "Code" keyword confuses the model. | Refinement. Future models need better separation of "Work Task" vs "Leisure". |

### Deep Dive: The Privacy vs. Utility Trade-off

A key limitation is distinguishing between work-related research (e.g., a medical writer researching "burnout") and personal distress. Our current model relies purely on semantic content. In production, this would require a "Human-in-the-loop" review of aggregated topics to ensure we don't misinterpret professional research as a cry for help.

---

## E. Deployment Architecture

### System Components

- **Streamlit Frontend:** Dashboard for HR Managers to view trends and topic distributions.
- **FastAPI Backend:** Processes logs in real-time and serves predictions.
- **Docker Compose:** Ensures secure and isolated deployment within the company intranet.