#!/bin/bash
#SBATCH --job-name=SDAL_DAL
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem=128GB
#SBATCH --array=0-719%4
#SBATCH --output=/mnt/stud/work/phahn/SDAL/logs/eval/%A_%a_%x.out

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
query_model=${models[$index % 3]}
s=${ssl[$index / 3 % 2]}
p=${pre[$index / 6 % 2]}
query=${queries[$index / 12 % 5]}
eval_model=${models[$index % 60 / 3]}
eval_s=${ssl[$index / 180 % 2]}
eval_p=${pre[$index / 360 % 2]}
seed=${random_seeds[$index / 720]}

# Predefine certain paths
dataset_path=/mnt/stud/work/phahn/datasets/
final_pool_dir=/mnt/stud/work/phahn/SDAL/storage/final_pools/${model}/${query}/ssl_${s}/pre_${p}/seed_${seed}/
output_dir=/mnt/stud/work/phahn/SDAL/output/eval_reu/${query_model}/${query}/ssl_${s}/pre_${p}/seed_${seed}/${eval_model}/ssl_${eval_s}/pre_${eval_p}/
model_dir=/mnt/stud/work/phahn/SDAL/storage/initial_weights/

# Run experiment
python -u evaluate_pool.py \
        path.output_dir=$output_dir \
        path.data_dir=$dataset_path \
        path.final_pool_dir=$final_pool_dir \
        path.model_dir=$model_dir \
        random_seed=$seed \
        model=$eval_model \
        ssl.use=$eval_s \
        pretrain.use=$eval_p \