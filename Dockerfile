# Use PyTorch base image with CUDA 12.1
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel

# Install system dependencies (including coinor-cbc for the MILP solver)
RUN apt-get update && apt-get install -y git wget gcc coinor-cbc && rm -rf /var/lib/apt/lists/*

# Set up non-root user (required by HF Spaces)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Install Unsloth & ML stack
RUN pip install --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
RUN pip install --no-cache-dir trl peft accelerate bitsandbytes

# Install Project dependencies
RUN pip install --no-cache-dir pulp pydantic fastapi uvicorn[standard] fastmcp gymnasium openenv-core trackio

# Copy application files
COPY --chown=user . /app/

# Start the FastAPI server (which kicks off training in the background)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
