# FSOD-TOPG

- **Modified from**
    - [*fundamentalvision/Deformable-DETR*](https://github.com/fundamentalvision/Deformable-DETR)
    - [*ZhangGongjie/Meta-DETR*](https://github.com/ZhangGongjie/Meta-DETR)

## User Guide

1. Log into your NYU HPC
2. Go to your scratch folder `cd /scratch/$USER`
3. Clone this repo and cd into it via `git clone https://github.com/NigelLu/FSOD-TOPG.git && cd FSOD-TOPG`
4. Request a GPU runtime via `srun --nodes=1 --cpus-per-task=1 --mem=32GB --time=2:00:00 --pty /bin/bash`
5. Start read-write singularity with COCO overlay `singularity exec --nv --bind /scratch/$USER --overlay /scratch/work/public/ml-datasets/coco/coco-2017.sqf --overlay /scratch/work/public/ml-datasets/coco/coco-2014.sqf --overlay /scratch/$USER/overlay-25GB-500K.ext3:rw /scratch/$USER/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash`
6. Run `make.sh` to enable some CUDA functionalities via `sh src/models/operations/make.sh`
7. Main training script can be accessed via `python -W ignore src/main.py --dataset_file coco_base --epochs 5`
    
    - Please make sure that you have enabled your python environment
    - If you have missing packages, please `conda activate <your_env> && python -m pip install --upgrade pip && python -m pip install -r requirements.txt`
8. To explore, you may import `pdb` at any place and use `pdb.set_trace()`