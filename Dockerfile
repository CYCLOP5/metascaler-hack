FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel

RUN apt-get update && apt-get install -y git wget gcc coinor-cbc && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

RUN pip install --no-cache-dir --upgrade "torchvision>=0.25.0" "torchaudio"
RUN pip install --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
RUN pip install --no-cache-dir trl peft accelerate bitsandbytes

RUN pip install --no-cache-dir pulp pydantic fastapi uvicorn[standard] fastmcp gymnasium openenv-core trackio

COPY --chown=user . /app/

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
