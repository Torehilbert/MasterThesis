#!/bin/sh
#BSUB -q gpuv100
#BSUB -J Inf
#BSUB -n 8
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 23:59
#BSUB -R "rusage[mem=3GB]"
#BSUB -u tore.hilbert@hotmail.com
#BSUB -B
#BSUB -N
#BSUB -o gpu-%J.out
#BSUB -e gpu_%J.err

# Load the modules
module load python3/3.7.5
module load cuda/10.0
module load cudnn/v7.6.5.32-prod-cuda-10.0

python3 ~/Speciale/Code/Inference/run_inference_multiple.py -data="/zhome/bc/0/98051/Speciale/Code/output/OGs" -test_data="/work1/s144328/test32"