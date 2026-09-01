#!/bin/bash
#SBATCH --job-name=fig5_galaxy
#SBATCH --account=bfir-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=/projects/bfir/ssourav/logs/fig5_%j.out
#SBATCH --error=/projects/bfir/ssourav/logs/fig5_%j.err

mkdir -p /projects/bfir/ssourav/logs
mkdir -p /projects/bfir/ssourav/exp4_results/figures

source /u/ssourav/miniforge/etc/profile.d/conda.sh
conda activate base

cd ~/cross-survey-conformal-prediction/adaptive-conformal

python make_galaxy_panels.py \
    --per_object /projects/bfir/ssourav/exp4_results/full/per_object_results.csv \
    --out_dir    /projects/bfir/ssourav/exp4_results/figures \
    --survey     legacysurvey_hsc \
    --backbone   astropt_850m \
    --n_per_cat  4
