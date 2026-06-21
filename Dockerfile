FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# These avoid a few cuDNN frontend / attention path issues and reduce GPU memory fragmentation.
ENV TORCH_CUDNN_V8_API_DISABLED=1
ENV CUDNN_FRONTEND_ATTN_DISABLED=1
ENV CUDA_MODULE_LOADING=LAZY
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3-dev \
    build-essential \
    git \
    curl \
    libgl1 \
    libglib2.0-0 \
    zlib1g \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel
RUN python -m pip install -r /app/requirements.txt
RUN python - <<'PY'
import site
from pathlib import Path

required = [
    "libcudnn.so.9",
    "libcudnn_adv.so.9",
    "libcudnn_cnn.so.9",
    "libcudnn_graph.so.9",
    "libcudnn_ops.so.9",
]

found_dirs = []
missing = []

for base in site.getsitepackages():
    d = Path(base) / "nvidia" / "cudnn" / "lib"
    if d.exists():
        found_dirs.append(str(d))
        print("cuDNN lib dir:", d)
        print("cuDNN libs:", sorted(p.name for p in d.glob("libcudnn*")))
        for name in required:
            if not (d / name).exists():
                missing.append(str(d / name))

if not found_dirs:
    raise SystemExit("No nvidia/cudnn/lib directory found")

if missing:
    raise SystemExit("Missing required cuDNN libs: " + ", ".join(missing))

print("cuDNN required libs OK")
PY
# For serverless without a network volume, this bakes the cache into the image.
# If you mount a RunPod network volume at /runpod-volume, it can override this path at runtime.
ENV HF_HOME=/runpod-volume/huggingface
ENV TRANSFORMERS_CACHE=/runpod-volume/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/runpod-volume/huggingface/sentence-transformers
ENV TORCH_HOME=/runpod-volume/torch
ENV XDG_CACHE_HOME=/runpod-volume/cache

RUN mkdir -p \
    /runpod-volume/huggingface \
    /runpod-volume/huggingface/sentence-transformers \
    /runpod-volume/torch \
    /runpod-volume/cache

RUN python - <<'PY'
import torch
import transformers
from transformers import AutoModel, AutoProcessor
print('torch:', torch.__version__)
print('torch cuda:', torch.version.cuda)
print('cuda available at build:', torch.cuda.is_available())
print('device count at build:', torch.cuda.device_count())
print('cudnn:', torch.backends.cudnn.version())
print('transformers:', transformers.__version__)
print('SigLIP 2 AutoModel/AutoProcessor imports OK')
PY

RUN python - <<'PY'
import onnxruntime as ort
print('onnxruntime:', ort.__version__)
print('device:', ort.get_device())
print('providers:', ort.get_available_providers())
assert 'CUDAExecutionProvider' in ort.get_available_providers(), 'CUDAExecutionProvider missing'
PY

# Pre-download buffalo_l. This uses CPU only during build; runtime enforces CUDAExecutionProvider.
RUN python - <<'PY'
from insightface.app import FaceAnalysis
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640, 640))
print('InsightFace buffalo_l downloaded')
PY

COPY handler.py face_dedup.py /app/
CMD ["python", "-u", "/app/handler.py"]
