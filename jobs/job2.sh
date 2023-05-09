#!/bin/bash

# env
source ~/.bashrc
conda activate LLM
cd /home/as14229/NYU_HPC/LLM-Personality-Codebase


# jobs

# python3 run.py --config 'OPT-6.7B' --order 'original'
# python3 run.py --config 'GPT2-Medium'
# python3 run.py --config 'GPT2-Large'
# python3 run.py --config 'GPTNEO-125M'

scancel $SLURM_JOB_ID