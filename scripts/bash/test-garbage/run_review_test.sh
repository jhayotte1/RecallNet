#!/bin/bash
#SBATCH --job-name=recallnet_70b
#SBATCH --gres=gpu:2
#SBATCH --time=23:45:00
#SBATCH --exclude=n51,n52,n53,n54,n55
#SBATCH --output=/mnt/beegfs/projects/RecallNet/src/results/logs/slurm_%j.out
#SBATCH --error=/mnt/beegfs/projects/RecallNet/src/results/logs/slurm_%j.err

source ~/.bashrc
conda activate recallnet

export RECALLNET_MODEL="llama3.3:70b"
export RECALLNET_BATCH_SIZE=10
export RECALLNET_MAX_CONCURRENCY=2
export RECALLNET_PROMPT="prompt_review.txt"

OLLAMA_FLASH_ATTENTION=1 OLLAMA_NUM_PARALLEL=2 ollama serve > ~/RecallNet/ollama.log 2>&1 &
sleep 5

cd /mnt/beegfs/projects/RecallNet/src/scripts
python3 LG_review.py \
    --exp-name exp05_LG_1k_top_100k

sleep 3
pkill ollama