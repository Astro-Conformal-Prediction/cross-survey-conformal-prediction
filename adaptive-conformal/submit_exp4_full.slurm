#!/bin/bash
#SBATCH --job-name=exp4_full
#SBATCH --account=bfir-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --time=08:00:00
#SBATCH --output=/projects/bfir/ssourav/logs/exp4_full_%j.out
#SBATCH --error=/projects/bfir/ssourav/logs/exp4_full_%j.err

source /u/ssourav/miniforge/etc/profile.d/conda.sh
conda activate base

cd ~/cross-survey-conformal-prediction/adaptive-conformal

python run_adaptive_cp.py \
    --dataset jwst_hsc legacysurvey_hsc sdss_hsc \
    --config astropt_850m astropt_95m vit_huge dino_small dino_giant \
    --coverage 80 90 95 \
    --difficulty_model hgbt ridge \
    --out_dir /projects/bfir/ssourav/exp4_results/full
