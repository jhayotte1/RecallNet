#!/bin/bash
#SBATCH --job-name=recallnet_70b
#SBATCH --gres=gpu:1
#SBATCH --time=23:00:00
#SBATCH --nodelist=n[1-5]
#SBATCH --output=/mnt/beegfs/projects/RecallNet/src/results/logs/slurm_%j.out
#SBATCH --error=/mnt/beegfs/projects/RecallNet/src/results/logs/slurm_%j.err

source ~/.bashrc
conda activate recallnet

OLLAMA_FLASH_ATTENTION=1 OLLAMA_NUM_PARALLEL=6 ollama serve > /tmp/ollama.log 2>&1 &
sleep 5

cd /mnt/beegfs/projects/RecallNet/src/scripts
python3 sample_dataset.py
python3 LG_classify.py

sleep 5
pkill ollama