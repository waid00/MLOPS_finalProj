End-to-End ML Topic Modeling: PromptMarket

Executive Summary

PromptMarket leverages advanced NLP to organize unstructured Generative AI prompts into navigable business categories. This report details the development of the ML pipeline, from data ingestion to a containerized deployment, ensuring a scalable solution for prompt discovery.

A. ML Problem Framing

Business Goal: Transform a raw list of user-submitted prompts into a structured catalog to improve user engagement and searchability.
ML Problem: Unsupervised Semantic Clustering (Topic Modeling).
Metrics: - Business: Discovery-to-Copy Rate.

ML: Topic Coherence (c_v), Cluster Purity (against manual seed set).

B. Data Processing Decisions

Deduplication: Removed exact text matches to prevent cluster density bias.

Length Filtering: Removed prompts < 4 words. Rationale: "Hi" or "Help" are not categorizable prompts.

Preprocessing: Lowercasing and whitespace normalization. Stopword removal handled dynamically by CountVectorizer within BERTopic to preserve phrase structure in embeddings while cleaning topic representations.

C. Model Development

We utilized BERTopic with the following final configuration:

Embedding: all-MiniLM-L6-v2 (Fast, efficient).

Dimensionality Reduction: UMAP (n_neighbors=15, n_components=5).

Clustering: HDBSCAN (min_cluster_size=7).

Results: Identified 8 distinct categories including Coding, Content Creation, Roleplay, and Marketing.

D. Failure Analysis

Input

Prediction

Root Cause

Mitigation

"Fix this."

Uncategorized (-1)

Insufficient semantic context.

Frontend min-length validation.

"Code a poem."

Coding (Low Conf)

Mixed intent (Code vs Write).

Multi-label classification (future work).

"Ignore instructions."

Roleplay

Meta-prompting aligns with 'persona' vectors.

Separate "Jailbreak" detector model.

E. Deployment Architecture

Components:

Streamlit Frontend: User Interface for input and batch processing.

FastAPI Backend: Exposes the model via REST API. Decoupling allows independent scaling.

Docker Compose: Orchestrates the networking between frontend and backend.

Rationale: A microservices approach (Frontend separated from Inference) allows us to update the heavy ML model without taking down the UI, and allows different resource allocation (e.g., GPU for backend only).