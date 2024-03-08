pip install "flash-attn>=2.5.0"  --no-build-isolation
pip install -e .
# export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=/mloscratch/homes/solergib/HF_CACHE
export CUDA_DEVICE_MAX_CONNECTIONS=1