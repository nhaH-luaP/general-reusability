#!/bin/bash
#SBATCH --job-name=SDAL_INIT
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem=128GB
#SBATCH --output=/mnt/stud/work/phahn/SDAL/logs/dal/initializing_%j.out
date;hostname;pwd
source /mnt/stud/work/phahn/venvs/sdal/bin/activate
cd /mnt/stud/work/phahn/SDAL/SDAL

dataset_path=/mnt/stud/work/phahn/datasets/
pool_dir=/mnt/stud/work/phahn/SDAL/storage/initial_pools/
model_dir=/mnt/stud/work/phahn/SDAL/storage/pretrained_models/

python -u initialize.py \
        path.initial_pool_dir=$pool_dir \
        path.data_dir=$dataset_path \
        path.model_dir=$model_dir \
        dataset=cifar10 \
	alc.initial_pool_size=300 \
        model=wideresnet282