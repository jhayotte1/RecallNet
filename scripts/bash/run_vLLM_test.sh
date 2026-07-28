#!/bin/bash
#SBATCH --job-name=recallnet_vLLM_test
#SBATCH --account=sta_inf
#SBATCH --partition=preemptable
#SBATCH --qos=preemptable
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpu-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/mnt/hdd/homes/jhayotte/RecallNet/src/results/logs/slurm_%j.out
#SBATCH --error=/mnt/hdd/homes/jhayotte/RecallNet/src/results/logs/slurm_%j.err

set -euo pipefail

