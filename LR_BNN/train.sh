
#####################CIFAR10
dataset='cifar10'
model_list=( 'dpn-92' 'vgg-19' 'resnet-101' 'densenet-121' 'densenet-169' 'preactresnet-18' 'resnext-29_8x16' 'preactresnet-164' 'wide-resnet-28x10' 'wide-resnet-40x10')
gpunumber=0
for model in "${model_list[@]}"
do

./launch_train_experiment.sh $dataset $model $gpunumber

done


