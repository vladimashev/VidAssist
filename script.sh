#!/bin/bash
#SBATCH --job-name=llama-example
#SBATCH --output=llama_out.log
#SBATCH --error=llama_err.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --export=ALL

python example.py