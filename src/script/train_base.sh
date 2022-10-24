#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu
#SBATCH --mem=200GB
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --cpus-per-task=2
#SBATCH --mail-user=xl3139@nyu.edu
#SBATCH --partition=a100,v100,rtx8000
#SBATCH --job-name=coco-base-deformable-detr

# * Singularity path
output_dir=$1
num_epochs=$2
batch_size=$3
ext3_path=/scratch/$USER/overlay-25GB-500K.ext3
sif_path=/scratch/$USER/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif

# * main script start
singularity exec --nv \
            --bind /scratch/$USER \
            --overlay /scratch/work/public/ml-datasets/coco/coco-2014.sqf \
            --overlay /scratch/work/public/ml-datasets/coco/coco-2017.sqf \
            --overlay ${ext3_path}:ro \
            ${sif_path} /bin/bash -c "source ~/.bashrc
            conda deactivate
            conda activate dl
            cd /scratch/$USER/FSOD-TOPG/src
            python main.py --output_dir ${output_dir} --coco_path /coco --dataset_file coco_base --epochs ${num_epochs} --batch_size ${batch_size}"