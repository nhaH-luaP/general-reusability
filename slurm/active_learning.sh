#!/bin/bash
#SBATCH --job-name=SDAL_DAL
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem=128GB
#SBATCH --array=0-59%4
#SBATCH --output=/mnt/stud/work/phahn/SDAL/logs/dal/%A_%a_%x.out

# Active Environment, change to directory and print certain infos
date;hostname;pwd
source /mnt/stud/work/phahn/venvs/sdal/bin/activate
cd /mnt/stud/work/phahn/SDAL/SDAL

# Create tupel of variables
models=(resnet10 resnet18 resnet34 resnet50)
queries=(random margin coreset badge typiclust)
random_seeds=(1 2 3)

# Get the current task index from the job array and select instances of variables based on it
index=$SLURM_ARRAY_TASK_ID
model=${models[$index % 4]}
query=${queries[$index / 4 % 5]}
seed=${random_seeds[$index / 20]}

# Predefine certain paths
dataset_path=/mnt/stud/work/phahn/datasets/
initial_pool_dir=/mnt/stud/work/phahn/SDAL/storage/initial_pools/
final_pool_dir=/mnt/stud/work/phahn/SDAL/storage/final_pools/${model}/${query}/seed_${seed}/
output_dir=/mnt/stud/work/phahn/SDAL/output/dal/${model}/${query}/seed_${seed}/
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
        model=$model
