#!/usr/bin/env bash
#SBATCH --job-name=outset_v2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --array=1-3
#SBATCH --output=%x_%a.out
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate /data/rauschecker1/jkw/envs/outset
export HF_TOKEN=$HF_TOKEN HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=3
cd "$HOME/outset-ai-healthcare"
case "$SLURM_ARRAY_TASK_ID" in
  1) python solutions/g2_skin_screening/run_experiment.py ;;
  2) python solutions/g4_skin_access/run_experiment.py ;;
  3) python solutions/g3_brain_ct/run_experiment.py ;;
esac
echo "task $SLURM_ARRAY_TASK_ID DONE"
