#!/bin/sh
#BSUB -q gpuv100
#BSUB -J DSCNet
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

python3 ~/Speciale/Code/DeepSubspaceClustering/train_dscnet.py -path_train_data HPC \
 -epochs=100 -batch_size=16 -optimizer=SGD -lr_start=0.1 -lr_steps 40 60 80 -lr_multiplier=0.1 \
 -encode_filters 16 32 -max_pool_strides 2 2 \
 -weight_decay_l2=0.0001 -weight_decay_coef_l2=0.0001 -alpha=1.0