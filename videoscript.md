Video Presentation Script

Part I: Business & ML Framing (0:00 - 5:00)

Speaker: "Hello, I am [Name]. Welcome to the PromptMarket presentation. We are addressing a critical issue in the GenAI ecosystem: Prompt Discovery."
Visual: Slide showing a chaotic list of CSV rows vs. a neat grid of categories.
Speaker: "Our business goal is to increase the 'Discovery-to-Copy' rate. Users shouldn't scroll through 1,000 lines to find a Python helper. They need a 'Coding' category."
Speaker: "We treated this as an Unsupervised Clustering problem. Using the ChatGPT Prompts dataset, we filtered for high-quality inputs and engineered features like prompt length to remove low-effort noise."

Part II: Demo (5:00 - 12:00)

Visual: Screen share of Streamlit App.
Speaker: "Let's look at the live app. Here is the 'Single Prompt' mode."
Action: Paste: "I want you to act as a social media manager for a coffee shop."
Speaker: "Click Classify. Result: 'Marketing & SEO'. Confidence: 85%. Correct."
Action: Paste: "Write a philosophical essay about the nature of time."
Speaker: "Result: 'Philosophy & Logic'. The model successfully distinguishes intent."
Action: Click 'Failure Analysis Demo'. Run 'Fix this.'
Speaker: "Here is a failure case. Short input 'Fix this' lacks context. The model returns 'Uncategorized' or a generic label. This informs us we need a minimum character limit on the frontend."

Part III: Technical Deep Dive (12:00 - 15:00)

Visual: VS Code / Architecture Diagram.
Speaker: "Under the hood, we use FastAPI serving a quantized BERTopic model. We chose all-MiniLM-L6-v2 for the embedding layer because it offers the best speed-to-performance ratio for CPU inference."
Visual: MLflow UI screenshot.
Speaker: "We ran 8 experiments. You can see here that increasing UMAP neighbors to 15 stabilized our clusters, whereas neighbors=5 fractured the 'Coding' topic into too many sub-fragments."
Speaker: "Thank you. This architecture is containerized and ready for cloud deployment."