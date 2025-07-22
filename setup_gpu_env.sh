#!/bin/bash
echo "=== 设置GPU环境变量 ==="

# 设置CUDA和cuDNN环境变量
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CUDA_HOME=$CONDA_PREFIX
export CUDA_ROOT=$CONDA_PREFIX

echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "CUDA_HOME: $CUDA_HOME"
echo "CUDA_ROOT: $CUDA_ROOT"

echo "=== 环境变量设置完成 ==="
echo "现在可以运行GPU检测脚本: python test_gpu_training.py"
echo "或者直接运行训练脚本: bash train.sh" 