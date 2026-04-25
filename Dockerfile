FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

RUN apt-get update && apt-get install -y git wget gcc coinor-cbc && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
ENV BNB_CUDA_VERSION=124
ENV LD_LIBRARY_PATH="/opt/conda/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}"

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip
# Keep HF Space runtime on a torch/transformers combo compatible with this CUDA base image.
RUN pip install --no-cache-dir \
  "transformers==4.49.0" \
  "accelerate==1.3.0" \
  "peft==0.14.0" \
  "datasets==3.2.0" \
  "trl==0.14.0" \
  "bitsandbytes==0.45.5"
RUN pip install --no-cache-dir "unsloth[cu124-torch250] @ git+https://github.com/unslothai/unsloth.git"
RUN pip uninstall -y torchao || true
RUN pip install --no-cache-dir pulp pydantic fastapi uvicorn[standard] fastmcp gymnasium openenv-core trackio

COPY --chown=user . /app/

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
