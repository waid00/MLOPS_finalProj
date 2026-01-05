FROM python:3.9-slim

WORKDIR /app

# 1. Install CPU-only PyTorch FIRST.
# This prevents pip from downloading the massive NVIDIA CUDA versions 
# when it encounters 'bertopic' or 'sentence-transformers' in requirements.txt.
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .

# 2. Install the rest of the dependencies. 
# Since torch is already installed (CPU version), pip will skip downloading the NVIDIA version.
RUN pip install --no-cache-dir -r requirements.txt

# Copy code and model
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]