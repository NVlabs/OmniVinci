#!/usr/bin/env bash
set -e

CONDA_ENV=${1:-""}
if [ -n "$CONDA_ENV" ]; then
    # This is required to activate conda environment
    eval "$(conda shell.bash hook)"

    conda create -n $CONDA_ENV python=3.10.14 -y
    conda activate $CONDA_ENV
    # This is optional if you prefer to use built-in nvcc
    conda install -c nvidia cuda-toolkit -y
else
    echo "Skipping conda environment creation. Make sure you have the correct environment activated."
fi

# Using uv to speedup installations
pip install uv
alias uvp="uv pip"

echo "[INFO] Using python $(which python)"
echo "[INFO] Using pip $(which pip)"
echo "[INFO] Using uv $(which uv)"

# This is required to enable PEP 660 support
uv pip install --upgrade pip setuptools

# Install FlashAttention2
uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Install VILA
uv pip install -e ".[train,eval]"

# numpy introduce a lot dependencies issues, separate from pyproject.yaml
# pip install numpy==1.26.4

# audio
uv pip install soundfile librosa openai-whisper ftfy
uv pip install ffmpeg
# conda install -c conda-forge ffmpeg
uv pip install jiwer

# Downgrade protobuf to 3.20 for backward compatibility
uv pip install protobuf==3.20.*

# Replace transformers and deepspeed files
site_pkg_path=$(python -c 'import site; print(site.getsitepackages()[0])')
cp -rv ./transformers/modeling_utils.py $site_pkg_path/transformers/modeling_utils.py # for using qwen 2.5 omni checkpoint

# for benchmark adoption
uv pip install faiss-gpu-cu12

# Quantization requires the newest triton version, and introduce dependency issue
uv pip install triton==3.1.0 # we don't need this version if we do not use FP8LinearQwen2Config, QLlavaLlamaConfig, etc. It is not compatible with mamba-ssm.

uv pip install kaldiio

# for rotary embedding
uv pip install beartype

uv pip install pydantic==1.10.22
