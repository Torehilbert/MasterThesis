#!/bin/sh
#BSUB -q gpuv100
#BSUB -J Autoenc_N0
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

python3 ~/Speciale/Code/Autoencoder/train_autoenc.py -path_train_data HPC -aug_noise_level=0.0