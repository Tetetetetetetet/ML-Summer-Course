export CUDA_VISIBLE_DEVICES=4,5,6,7
model="ResNet"
mode="network_data"
k=all
feature_selector=None
resnet_gpu_batch_size=64 # 更新为最佳批次大小
resnet_epochs=200
isoversample=false

if [ $isoversample == true ]; then
    exp_name="${mode}_${k}_${model}_${feature_selector}_oversample"
    python src/data_fit.py --mode $mode -e $exp_name -k $k --model $model --isoversample --feature_selector $feature_selector
else
    if [ $model == "ResNet" ]; then
        exp_name="${mode}-k_${k}-model_${model}-bs_${resnet_gpu_batch_size}-ep_${resnet_epochs}-fs_${feature_selector}-notoversample"
        python src/data_fit.py --mode $mode -e $exp_name -k $k --model $model --feature_selector $feature_selector --resnet_gpu_batch_size $resnet_gpu_batch_size --resnet_epochs $resnet_epochs
    else
        exp_name="${mode}-k_${k}-model_${model}-fs_${feature_selector}-notoversample"
        python src/data_fit.py --mode $mode -e $exp_name -k $k --model $model --feature_selector $feature_selector
    fi
fi