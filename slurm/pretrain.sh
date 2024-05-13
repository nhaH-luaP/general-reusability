#!/bin/bash
#SBATCH --job-name=SDAL_PRE
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem=128GB
#SBATCH --array=0-2%4
#SBATCH --output=/mnt/stud/work/phahn/SDAL/logs/pre/%A_%a_%x.out

# Active Environment, change to directory and print certain infos
date;hostname;pwd
source /mnt/stud/work/phahn/venvs/sdal/bin/activate
cd /mnt/stud/work/phahn/SDAL/SDAL

# Create tupel of variables
models=(wideresnet2810)
random_seeds=(1 2 3)

# Get the current task index from the job array and select instances of variables based on it
index=$SLURM_ARRAY_TASK_ID
model=${models[$index % 3]}
seed=${random_seeds[$index / 3]}

# Predefine certain paths
dataset_path=/mnt/stud/work/phahn/datasets/
output_dir=/mnt/stud/work/phahn/SDAL/output/pre/${model}/seed_${seed}/
model_dir=/mnt/stud/work/phahn/SDAL/storage/initial_weights/pretrained/

# Run experiment
python -u pretrain.py \
        path.output_dir=$output_dir \
        path.data_dir=$dataset_path \
        path.model_dir=$model_dir \
        random_seed=$seed \
        model=$model