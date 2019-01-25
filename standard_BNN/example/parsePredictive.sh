database='cifar10'
modelall=( 'preactresnet-164')

for model in "${modelall[@]}"
do


dirPredictive="./predictiveResult/"$database"/"$model"/"

echo "Neuronas por capa,capas,MonteCarlo,Epochs,DKLSF,ECE15,Accuracy,ECE15Valid,MCValid" >   "ParsePredictive"$database$model".txt"

for file in `ls $dirPredictive`
do

topology=`echo $file | cut -d"-" -f1`
numero_capas=`echo $topology | grep -o "_" | wc -l`
numero_capas="$(($numero_capas -1))"
neuronas_por_capa=`echo $topology | cut -d"_" -f2`

algo_params=`echo $file | cut -d"-" -f2`
mc_samples=`echo $algo_params | cut -d "_" -f1 | cut -d "M" -f1 `
epochs=`echo $algo_params | cut -d "_" -f2 | cut -d "e" -f1 `
dklsf=`echo $algo_params | cut -d "_" -f3 | cut -d "D" -f1 `


cat $dirPredictive"/"$file  | grep "Best" | awk -v dklsf="$dklsf" -v var="$file" -v mc_samples="$mc_samples" -v epochs="$epochs" -v num_capas="$numero_capas" -v neuronas_por_capa="$neuronas_por_capa" 'BEGIN{}{print neuronas_por_capa","num_capas","mc_samples","epochs","dklsf", , ,"$7","$10} ' >> "ParsePredictive"$database$model".txt"


done
done
