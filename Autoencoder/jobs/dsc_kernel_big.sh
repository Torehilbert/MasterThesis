#!/bin/sh
#BSUB -q gpuv100
#BSUB -J DSCNet_Large
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 23:59
#BSUB -R "rusage[mem=4GB]"
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

python3 ~/Speciale/CodeGit/Autoencoder/train_autoenc.py -path_train_data HPC -encoder_kernel_sizes 5 3 -encoder_filters 32 64 -decoder_filters 64 32 -decoder_kernel_sizes 3 5 -improvement_patience=10 -learning_rate_initial=0.00005 -batch_size=32