#!/bin/bash
#SBATCH --job-name=recallnet_filter
#SBATCH --gres=gpu:1
#SBATCH --time=23:30:00
#SBATCH --output=/mnt/beegfs/projects/RecallNet/src/results/logs/slurm_%j.out
#SBATCH --error=/mnt/beegfs/projects/RecallNet/src/results/logs/slurm_%j.err

source ~/.bashrc
conda activate recallnet

export RECALLNET_MODEL="llama3.1:8b"
export RECALLNET_BATCH_SIZE=10
export RECALLNET_MAX_CONCURRENCY=6
export RECALLNET_PROMPT="prompt_filter.txt"


OLLAMA_FLASH_ATTENTION=1 OLLAMA_NUM_PARALLEL=6 ollama serve > /tmp/ollama.log 2>&1 &
sleep 5

cd /mnt/beegfs/projects/RecallNet/src/scripts
python3 LG_filter.py \
    --split-dir "1_SPLITED" \
    --exp-name exp05_LG_1k_top_100k \
    --sample-size 100

sleep 3
pkill ollama