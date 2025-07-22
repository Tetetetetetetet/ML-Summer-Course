from src.data_fit import DataFit
import pdb
from argparse import Namespace
args = Namespace(
    mode='network_data',           # 数据集模式: 'normal', 'selected', 'network_data', '2class'
    isoversample=True,            # 是否使用过采样
    exp_name='report',      # 实验名称
    k='all',                       # 特征选择数量: 'all' 或整数
    model='ResNet',                # 模型名称: 'all', 'ResNet', 'RandomForest', 'GradientBoosting', 'LogisticRegression'
    feature_selector="None",         # 特征选择方法: 'chi2', 'f_classif', None
    resnet_gpu_batch_size=128,     # ResNet GPU批次大小
    resnet_epochs=1000,             # ResNet训练轮数
    cover_old_result=False,         # 是否覆盖旧的实验结果
    eval=True
)
data_fit = DataFit(args=args)
data_fit.run_complete_pipeline()