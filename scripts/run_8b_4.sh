#!/bin/bash
#SBATCH --job-name=recallnet_70b
#SBATCH --gres=gpu:1
#SBATCH --time=23:00:00
#SBATCH --exclude=n51,n52,n53,n54,n55,n101,n102
#SBATCH --output=/mnt/beegfs/projects/RecallNet/src/results/logs/slurm_%j.out
#SBATCH --error=/mnt/beegfs/projects/RecallNet/src/results/logs/slurm_%j.err

source ~/.bashrc
conda activate recallnet

export RECALLNET_MODEL="llama3.1:8b"
export RECALLNET_BATCH_SIZE=10
export RECALLNET_MAX_CONCURRENCY=10

OLLAMA_FLASH_ATTENTION=1 OLLAMA_NUM_PARALLEL=10 ollama serve > /tmp/ollama.log 2>&1 &
sleep 5

PREDS="hasprerequisite createdby motivatedbygoal hasfirstsubevent"

cd /mnt/beegfs/projects/RecallNet/src/scripts
python3 LG_classify.py \
    --data-dir ../data/sample_data_1k_top_100k \
    --exp-name exp07_LG \
    --exp-desc "8B scoring with scope definitions"

sleep 3
pkill ollama