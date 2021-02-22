#!/bin/sh
#BSUB -q gpuv100
#BSUB -J C2345
#BSUB -n 8
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 23:59
#BSUB -R "rusage[mem=3GB]"
#BSUB -u tore.hilbert@hotmail.com
#BSUB -B
#BSUB -N
#BSUB -o gpu-%J.out
#BSUB -e gpu_%J.err

module load python3/3.7.5
module load cuda/10.0
module load cudnn/v7.6.5.32-prod-cuda-10.0

python3 ~/Speciale/Code/sessions/train_manual.py -path_training_data HPC -epochs=120 -lr_steps 50 70 90 -aug_noise=0.25 -weight_decay_l2=0.0001 -resnet_n_channels=4 -use_channels 2 3 4 5 -resnet_block_sizes 3 5 3 -resnet_filter_size 32 64 128