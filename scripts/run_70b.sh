#!/bin/bash
#SBATCH --job-name=recallnet_70b
#SBATCH --gres=gpu:2
#SBATCH --time=10:00:00
#SBATCH --output=/mnt/beegfs/projects/RecallNet/src/results/slurm_%j.out
#SBATCH --error=/mnt/beegfs/projects/RecallNet/src/results/slurm_%j.err

source ~/.bashrc
conda activate recallnet

ollama serve > /tmp/ollama.log 2>&1 &
sleep 5

cd /mnt/beegfs/projects/RecallNet/src/scripts
python3 LG_classify.py

sleep 5
pkill ollama