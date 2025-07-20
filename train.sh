model="LogisticRegression"
mode="normal"
k=all
feature_selector="chi2"
isoversample=false
if [ $isoversample == true ]; then
    exp_name="${mode}_${k}_${model}_${feature_selector}_oversample"
    python src/data_fit.py --mode $mode -e $exp_name -k $k --model $model --isoversample --feature_selector $feature_selector
else
    exp_name="${mode}_${k}_${model}_${feature_selector}_notoversample"
    python src/data_fit.py --mode $mode -e $exp_name -k $k --model $model --feature_selector $feature_selector
fi