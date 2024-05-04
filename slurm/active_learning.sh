#!/bin/bash
#SBATCH --job-name=SDAL_DAL
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem=128GB
#SBATCH --array=0-179%4
#SBATCH --output=/mnt/stud/work/phahn/SDAL/logs/dal/%A_%a_%x.out

# Active Environment, change to directory and print certain infos
date;hostname;pwd
source /mnt/stud/work/phahn/venvs/sdal/bin/activate
cd /mnt/stud/work/phahn/SDAL/SDAL

# Create tupel of variables
models=(resnet6 resnet10 resnet18)
queries=(random margin coreset badge typiclust)
ssl=(False True)
pre=(False True)
random_seeds=(1 2 3)

# Get the current task index from the job array and select instances of variables based on it
index=$SLURM_ARRAY_TASK_ID
model=${models[$index % 3]}
s=${ssl[$index / 3 % 2]}
p=${pre[$index / 6 % 2]}
query=${queries[$index / 12 % 5]}
seed=${random_seeds[$index / 60]}

# Predefine certain paths
dataset_path=/mnt/stud/work/phahn/datasets/
initial_pool_dir=/mnt/stud/work/phahn/SDAL/storage/initial_pools/
final_pool_dir=/mnt/stud/work/phahn/SDAL/storage/final_pools/${model}/${query}/ssl_${s}/pre_${p}/seed_${seed}/
output_dir=/mnt/stud/work/phahn/SDAL/output/dal/${model}/${query}/ssl_${s}/pre_${p}/seed_${seed}/
model_dir=/mnt/stud/work/phahn/SDAL/storage/initial_weights/

# Run experiment
python -u active_learning.py \
        path.output_dir=$output_dir \
        path.data_dir=$dataset_path \
        path.initial_pool_dir=$initial_pool_dir \
        path.final_pool_dir=$final_pool_dir \
        path.model_dir=$model_dir \
        random_seed=$seed \
        al.query_strategy=$query \
        model=$model \
        ssl.use=$s \
        pretrain.use=$p 