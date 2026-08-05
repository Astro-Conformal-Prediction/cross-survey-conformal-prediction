#!/bin/bash
#SBATCH --job-name=exp4_adaptive_cp
#SBATCH --account=bfir-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/projects/bfir/ssourav/logs/exp4_%j.out
#SBATCH --error=/projects/bfir/ssourav/logs/exp4_%j.err

mkdir -p /projects/bfir/ssourav/logs

source /u/ssourav/miniforge/etc/profile.d/conda.sh
conda activate base   # or your usual env

cd ~/cross-survey-conformal-prediction/adaptive-conformal

# Step 1 of spec: verify on AstroPT-850M + HSC->JWST at 90% first
python run_adaptive_cp.py \
    --dataset jwst_hsc \
    --config astropt_850m \
    --coverage 90 \
    --difficulty_model hgbt \
    --out_dir /projects/bfir/ssourav/exp4_results/jwst_hsc
