#!/bin/bash

echo "=== 开始GPU训练 ==="

# 设置GPU环境变量
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CUDA_HOME=$CONDA_PREFIX
export CUDA_ROOT=$CONDA_PREFIX

echo "环境变量设置完成:"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "CUDA_HOME: $CUDA_HOME"
echo "CUDA_ROOT: $CUDA_ROOT"

# 设置可见的GPU设备（选择相对空闲的GPU）
export CUDA_VISIBLE_DEVICES=3,4,6,7
echo "使用GPU设备: $CUDA_VISIBLE_DEVICES"

# 检查GPU状态
echo "检查GPU状态..."
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | grep -E "(3|4|5|6)"

echo "=== 开始训练 ==="

model="ResNet"
# mode="selected"
mode="LR_pipeline"
# mode="one_hot"
k=all
feature_selector=None
resnet_epochs=1000
isoversample=true
resnet_batch_size=128
resnet_gpu_batch_size=1024 # 更新为最佳批次大小
eval=False
cover_old_result=True

echo "train.sh"
cat train.sh

if [ $isoversample == true ]; then
    if [ $model == "ResNet" ]; then
        exp_name="${mode}-k_${k}-model_${model}-bs_${resnet_gpu_batch_size}-ep_${resnet_epochs}-fs_${feature_selector}-oversample"
        python src/data_fit.py --mode $mode -e $exp_name -k $k --model $model --isoversample --feature_selector $feature_selector --resnet_batch_size $resnet_batch_size --resnet_gpu_batch_size $resnet_gpu_batch_size --resnet_epochs $resnet_epochs --eval $eval --cover_old_result $cover_old_result
    else
        exp_name="${mode}-k_${k}-model_${model}-fs_${feature_selector}-oversample"
        python src/data_fit.py --mode $mode -e $exp_name -k $k --model $model --isoversample --feature_selector $feature_selector --eval $eval --cover_old_result $cover_old_result
    fi
else
    if [ $model == "ResNet" ]; then
        exp_name="${mode}-k_${k}-model_${model}-bs_${resnet_gpu_batch_size}-ep_${resnet_epochs}-fs_${feature_selector}-notoversample"
        python src/data_fit.py --mode $mode -e $exp_name -k $k --model $model --feature_selector $feature_selector --resnet_batch_size $resnet_batch_size --resnet_gpu_batch_size $resnet_gpu_batch_size --resnet_epochs $resnet_epochs --eval $eval --cover_old_result $cover_old_result
    else
        exp_name="${mode}-k_${k}-model_${model}-fs_${feature_selector}-notoversample"
        python src/data_fit.py --mode $mode -e $exp_name -k $k --model $model --feature_selector $feature_selector --eval $eval --cover_old_result $cover_old_result
    fi
fi

echo "=== 训练完成 ==="
echo "结果保存在: output/${exp_name}/"