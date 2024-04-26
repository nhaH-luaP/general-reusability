#!/bin/bash
#SBATCH --job-name=SDAL_DAL
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem=128GB
#SBATCH --array=0-59%4
#SBATCH --output=/mnt/stud/work/phahn/SDAL/logs/eval/%A_%a_%x.out

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
query_model=${models[$index % 4]}
query=${queries[$index / 4 % 5]}
seed=${random_seeds[$index / 20]}
eval_model=wideresnet2810

# Predefine certain paths
dataset_path=/mnt/stud/work/phahn/datasets/
final_pool_dir=/mnt/stud/work/phahn/SDAL/storage/final_pools/${query_model}/${query}/seed_${seed}/
output_dir=/mnt/stud/work/phahn/SDAL/output/eval_fut/${eval_model}/${query_model}/${query}/seed_${seed}/
model_dir=/mnt/stud/work/phahn/SDAL/storage/initial_weights/

# Run experiment
python -u evaluate_pool.py \
        path.output_dir=$output_dir \
        path.data_dir=$dataset_path \
        path.final_pool_dir=$final_pool_dir \
        path.model_dir=$model_dir \
        random_seed=$seed \
        model=$eval_model
