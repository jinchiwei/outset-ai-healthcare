#!/usr/bin/env bash
#SBATCH --job-name=outset_sol
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --array=1-3
#SBATCH --output=%x_%a.out

# Runs the three GPU solution experiments in parallel (one per array task).
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate /data/rauschecker1/jkw/envs/outset
export HF_TOKEN=$HF_TOKEN
export HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=2

REPO="$HOME/outset-ai-healthcare"
case "$SLURM_ARRAY_TASK_ID" in
  1) SCRIPT="solutions/g2_skin_screening/run_experiment.py" ;;
  2) SCRIPT="solutions/g3_brain_ct/run_experiment.py" ;;
  3) SCRIPT="solutions/g4_skin_access/run_experiment.py" ;;
esac
echo "task $SLURM_ARRAY_TASK_ID -> $SCRIPT  on $(hostname)  gpu=$CUDA_VISIBLE_DEVICES"
cd "$REPO"
python "$SCRIPT"
echo "task $SLURM_ARRAY_TASK_ID DONE"
