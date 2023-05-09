#!/bin/bash

# env
source ~/.bashrc
conda activate LLM
cd /home/as14229/NYU_HPC/LLM-Personality-Codebase


# jobs

python3 run.py --config 'OPT-6.7B' --order 'reverse'
python3 run.py --config 'OPT-6.7B' --order 'order-III'
python3 run.py --config 'OPT-1.3B'
python3 run.py --config 'GPT2'



scancel $SLURM_JOB_ID