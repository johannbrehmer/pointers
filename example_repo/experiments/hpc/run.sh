#!/usr/bin/env bash

#SBATCH --job-name=imagenet
#SBATCH --output="log_imagenet_%a.log"
#SBATCH --error="log_imagenet_%a.err"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=5-00:00:00
#SBATCH --gres=gpu:1

module load cuda/10.1.105
conda activate crow

dir=/path/to/experiment/directory
seed=$((SLURM_ARRAY_TASK_ID + 129363))
setup=$((SLURM_ARRAY_TASK_ID))

# Cross-checking the installation and setup
which python
nvcc --version
nvidia-smi

cd $dir

case ${setup} in
0) python images.py with config1 seed=$seed;;
1) python images.py with config2 seed=$seed;;
*) echo "Nothing to do for job ${SLURM_ARRAY_TASK_ID}" ;;
esac
