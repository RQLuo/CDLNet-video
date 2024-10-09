#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --account=pr_198_general
#SBATCH --mem=64GB
#SBATCH --time=5:00:00
#SBATCH --job-name=single_keen_mcsure_169_30_7_2_16

python train3d.py args3dmri.json
