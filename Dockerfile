FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

RUN apt-get update && apt-get install -y git wget gcc g++ coinor-cbc libstdc++6 && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
ENV BNB_CUDA_VERSION=124
ENV LD_LIBRARY_PATH="/opt/conda/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}"

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir --force-reinstall \
  "torch==2.5.1" \
  "torchvision==0.20.1" \
  "torchaudio==2.5.1" \
  --index-url https://download.pytorch.org/whl/cu124
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
RUN python - <<'PY'
import torch, bitsandbytes
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("bitsandbytes", bitsandbytes.__version__)
assert str(torch.version.cuda).startswith("12.4"), f"unexpected torch CUDA: {torch.version.cuda}"
assert bitsandbytes.__version__ == "0.45.5", f"unexpected bitsandbytes: {bitsandbytes.__version__}"
PY

COPY --chown=user . /app/

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
