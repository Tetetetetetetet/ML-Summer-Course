.PHONY: all pipeline process impute clean

all: pipeline

pipeline: impute

# 1. 数据预处理 - 训练集
Dataset/processed/train_processed/recoded_train.csv:
	python src/data_process.py

# 1. 数据预处理 - 测试集
Dataset/processed/test_processed/recoded_test.csv:
	python src/data_process_test.py

# 2. 缺失值填充
Dataset/processed/train_processed/logistic_imputed/logistic_imputed_train_final.csv Dataset/processed/train_processed/logistic_imputed/logistic_imputed_test_final.csv: \
	Dataset/processed/train_processed/recoded_train.csv Dataset/processed/test_processed/recoded_test.csv
	python src/logistic_imputation_pipeline.py

impute: Dataset/processed/train_processed/logistic_imputed/logistic_imputed_train_final.csv Dataset/processed/train_processed/logistic_imputed/logistic_imputed_test_final.csv

check:
	python src/test_recoded_data.py

process: Dataset/processed/train_processed/recoded_train.csv Dataset/processed/test_processed/recoded_test.csv

clean:
	rm -rf Dataset/processed/train_processed/logistic_imputed/*
	rm -f Dataset/processed/train_processed/recoded_train.csv
	rm -f Dataset/processed/test_processed/recoded_test.csv