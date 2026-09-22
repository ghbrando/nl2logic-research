# Spark-compatible starting point from NVIDIA's PyTorch fine-tuning playbook.
# For a frozen experiment, override with a verified image digest.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.11-py3@sha256:417cbf33f87b5378849df37983552cd1f8bc8b62fe1ceabe004de816a55dff21
FROM ${BASE_IMAGE}
USER root
WORKDIR /workspace
COPY containers/requirements.txt /tmp/container-requirements.txt
# Preserve the NVIDIA image's CUDA-enabled torch build during dependency resolution.
RUN python -c "import torch; print('torch==' + torch.__version__)" > /tmp/torch-constraint.txt \
    && python -m pip install --no-cache-dir -c /tmp/torch-constraint.txt \
       -r /tmp/container-requirements.txt \
    && python -m pip check \
    && python -m pip freeze > /opt/nl2logic-packages.txt
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HOME=/tmp/home HF_HOME=/cache/huggingface XDG_CACHE_HOME=/cache
USER 1000:1000
ENTRYPOINT []
CMD ["python", "--version"]
