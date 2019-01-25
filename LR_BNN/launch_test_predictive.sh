database='putyourdatabasehere'
model='putyourmodelhere'
inputdim='putdimensions'

dir="./pretrain_models/"$database"/"$model"/"

dir="../RESULTADOS_ICML/GENDER/modelos/"$model"/"

#only for small files if not parse by line
file_head=`cat "./ParsePredictive"$database$model".txt"  | awk 'BEGIN{}{print $0;exit}'`
file=`cat "./ParsePredictive"$database$model".txt"  | awk 'BEGIN{}{if (NR>1){print $0}}'`

echo $file_head > "TestPredictive"$database$model".txt" 
IFS=','
while read neuron capas mcsamplestrain epochs dklsf  null1 null2  ecevalid mcvalid ; do

if [ "$capas" -eq 2 ]; then
top=$neuron"_"$neuron
elif [ "$capas" -eq 3 ]; then
top=$neuron"_"$neuron"_"$neuron
else
top=$neuron
fi

model_dir=$dir$inputdim"_"$top"_"$inputdim

var=`python ../predictive_estimation.py --model_net $model --data_dir ../data/ --dataset $database --n_gpu 0 --valid_test test --MCsamples $mcvalid --n_layers $capas --layer_dim $neuron --model_dir $model_dir"/"$mcsamplestrain"MC_"$epochs"eps_DKLSF"$dklsf"/models/BNN.pth.tar" |  awk 'BEGIN{}{print $11","$15","substr($18,1,length($18)-1)}'`

read ece _ acc <<< "$var"


echo $neuron","$capas","$mcsamplestrain","$epochs","$dklsf","$ece","$acc","$ecevalid","$mcvalid >> "TestPredictive"$database$model".txt"


done  <<< "$file"
