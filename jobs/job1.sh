#!/bin/bash

# env
source ~/.bashrc
conda activate LLM
cd /home/as14229/NYU_HPC/LLM-Personality-Codebase


# jobs
python3 run.py --config 'OPT-13B' --order 'original'
python3 run.py --config 'OPT-13B' --order 'order-III'
python3 run.py --config 'OPT-1.3B'
python3 run.py --config 'GPT2-Large'
python3 run.py --config 'GPTNEO-1.3B'


scancel $SLURM_JOB_ID