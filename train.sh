model="all"
mode="selected"
k=all
isoversample=false
if [ $isoversample == true ]; then
    exp_name="${mode}_${k}_${model}_oversample"
    python src/data_fit.py --mode $mode -e $exp_name -k $k --model $model --isoversample
else
    exp_name="${mode}_${k}_${model}_notoversample"
    python src/data_fit.py --mode $mode -e $exp_name -k $k --model $model
fi