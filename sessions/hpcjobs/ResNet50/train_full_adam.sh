#!/bin/sh
#BSUB -q gpuv100
#BSUB -J ResNet50
#BSUB -n 8
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 23:59
#BSUB -R "rusage[mem=3GB]"
#BSUB -u tore.hilbert@hotmail.com
#BSUB -B
#BSUB -N
#BSUB -o gpu-%J.out
#BSUB -e gpu_%J.err

##nvidia-smi

# Load the modules
module load python3/3.7.5
module load cuda/10.0
module load cudnn/v7.6.5.32-prod-cuda-10.0

python3 ~/Speciale/Code/sessions/train_manual.py -path_training_data HPC \
 -epochs=120 -optimizer=Adam -lr_start=0.001 lr_steps 50 70 90 -lr_multiplier=1.0 -aug_noise=0.25 -weight_decay_l2=0.0001 \
 -resnet_n_channels=5 -use_channels 1 3 4 6 7 -resnet_block_sizes 4 4 5 1 -resnet_filter_size 64 128 256 512 \
 -resnet_size_reductions 0 1 1 1 -resnet_stem_params 64 7 2 -resnet_stem_max_pool=1 -resnet_stem_max_pool_size=3