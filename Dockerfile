FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

RUN apt-get update && apt-get install -y git wget gcc coinor-cbc && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir "unsloth @ git+https://github.com/unslothai/unsloth.git"
RUN pip install --no-cache-dir trl peft accelerate bitsandbytes
RUN pip install --no-cache-dir pulp pydantic fastapi uvicorn[standard] fastmcp gymnasium openenv-core trackio

COPY --chown=user . /app/

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
