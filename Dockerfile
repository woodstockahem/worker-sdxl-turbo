FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Install Python 3.11 and necessary system tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
RUN pip install uv

# Create the virtual environment
RUN uv venv --python 3.11 /.venv

# Add the virtual environment to the PATH so 'python' automatically uses it
ENV PATH="/.venv/bin:$PATH"

# Install Python packages (xformers is unpinned to fix the torch.xpu error)
RUN uv pip install torch --extra-index-url https://download.pytorch.org/whl/cu121 \
    diffusers transformers accelerate safetensors xformers runpod numpy==1.26.3 scipy triton huggingface-hub hf_transfer setuptools Pillow

# Copy project files into the container
COPY download_weights.py schemas.py handler.py test_input.json /

# Download the weights from Hugging Face
RUN python /download_weights.py

# Run the handler
CMD ["python", "/handler.py"]
