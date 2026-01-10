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

This section provides a critical analysis of the *Employee Insight Engine* performance. The goal is not only to identify errors but to understand their origins and establish safe boundaries for deployment. In the context of HR and mental health, understanding these limits is essential to prevent false alarms or missed risk signals.

### 1. Failure Cases Analysis

The following table documents specific instances where the hybrid model (BERTopic + Keyword Override) failed to correctly identify user intent, along with mitigation strategies.

| Input Prompt | Model Prediction | Root Cause | HR Impact & Mitigation |
| :--- | :--- | :--- | :--- |
| **"Fix this."** | *Technical & Coding (Low Confidence)* | **Lack of Context:** The imperative "fix" is strongly correlated with code in training data. However, without an object, it's impossible to determine if it refers to code, a printer, or a process error. | **Threshold Filtering:** The system must ignore predictions with probability < 85%. HR should not be overwhelmed by vague complaints lacking a clear target. |
| **"Write a poem about flowers in Python style."** | *Technical & Coding* | **Keyword Bias:** The rule-based system detected "Python" and overrode the semantic meaning of the rest of the sentence. The model ignored the creative/leisure intent. | **Finer Segmentation:** Future versions must distinguish "Work-related coding" from "Leisure/Creative tasks". False classification here skews team productivity metrics. |
| **"I am struggling with the new manager's leadership."** | *General Work Task* | **Semantic Blindness to Sentiment:** The model correctly identified the entity "manager" but failed to capture the negative sentiment and interpersonal conflict belonging to Wellbeing. | **Sentiment Layer:** A parallel sentiment analysis model is required. Topic modeling alone cannot reliably detect frustration without explicit keywords. |
| **"The server is on fire, help!"** | *Employee Wellbeing (Flagged)* | **Metaphor Misinterpretation:** Keywords "fire" and "help" triggered a crisis scenario for burnout/safety, although it was a technical urgency (IT Incident). | **Contextual Disambiguation:** The model requires training on specific corporate slang to distinguish technical "firefighting" from actual psychological distress. |

### 2. Root Cause Analysis

Why do these errors occur? Our analysis revealed three main factors:

1.  **Hybrid Logic Conflict:** Our approach combines unsupervised learning (BERTopic) with hard rules (dictionaries). In edge cases like sarcasm or metaphors, keywords take absolute precedence. This increases "Recall" (we catch all Python mentions) but decreases "Precision" (we catch things unrelated to programming).
2.  **Training Data Limitations:** The model was trained on the *Awesome-ChatGPT-Prompts* dataset, which predominantly contains "Act as..." instructions. Real corporate communication is much more fragmented, informal, and contains grammatical errors, which the model is not fully adapted to.
3.  **Absence of Temporal Context:** The model classifies each prompt in isolation. It cannot distinguish between a one-off complaint ("I'm tired of this meeting") and a chronic issue that should concern HR.

### 3. Limitations & Privacy Trade-off

The most fundamental limitation of the current solution is the ability to distinguish between **work-related research** and **personal distress**.
* *Example:* If a medical writer researches "symptoms of burnout" for an article, our model might incorrectly classify this as a cry for help (Wellbeing Flag).
* *Consequence:* This is a fundamental limitation of semantic analysis lacking access to employee "roles." In production, this would require a **"Human-in-the-loop"** implementation, where aggregated data is validated before any intervention to ensure employee trust is not breached.

### 4. Edge Cases

The model shows instability with the following input types:
* **Multi-intent Prompts:** Inputs combining multiple requests (e.g., *"Check this code and then book me a vacation"*) end up in a random category based on which word carries more weight in the embedding space.
* **Code Injection:** If a user pastes a large block of logs or code (50+ lines) with a short question at the end, the model gets overwhelmed by technical noise, and the semantic meaning of the question is lost in the embedding.
* **Negation:** Sentences like *"I don't want to deal with HR"* might paradoxically be classified into the HR category because the model reacts to the presence of keywords, not their negation.

### 5. Future Improvements

With more time and resources, we would propose the following optimizations:
* **Hierarchical Modeling:** Instead of flat classification into 8 topics, we would introduce a two-stage system. Stage 1 would determine the department (IT/HR/Sales), and Stage 2 the specific issue.
* **Active Learning Loop:** Implementation of a mechanism where HR managers could (anonymously) flag incorrectly classified topics. This feedback would serve to continuously retrain the model and update the keyword list.
* **Sentiment Analysis Integration:** Adding a separate model (e.g., RoBERTa fine-tuned for sentiment) whose output would serve as a weight for the final category decision, especially for the *Employee Wellbeing* topic.

---

## E. Deployment Architecture

### System Components

- **Streamlit Frontend:** Dashboard for HR Managers to view trends and topic distributions.
- **FastAPI Backend:** Processes logs in real-time and serves predictions.
- **Docker Compose:** Ensures secure and isolated deployment within the company intranet.

![Architecture Diagram](assets/architecture.png)
