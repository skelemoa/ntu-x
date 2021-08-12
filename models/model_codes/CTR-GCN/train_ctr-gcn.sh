#!/bin/bash
#SBATCH -w gnode86
#SBATCH -A anirudh.thatipelli
#SBATCH -c 40
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2048
#SBATCH --time=3-00:00:00
##SBATCH --mail-type=ALL


source ~/env_2s_agcn/bin/activate

module load python/3.6.8
python3 main.py --config config/nturgbd120-cross-subject/default.yaml