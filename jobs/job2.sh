#!/bin/bash

# env
source ~/.bashrc
conda activate LLM
cd /home/as14229/NYU_HPC/LLM-Personality-Codebase


# jobs
python3 run.py --config 'OPT-13B' --order 'reverse'
python3 run.py --config 'OPT-6.7B' --order 'original'
python3 run.py --config 'OPT-6.7B' --order 'order-II'
python3 run.py --config 'GPT2-Medium'
python3 run.py --config 'GPTNEO-125M'
python3 run.py --config 'OPT-125M'


scancel $SLURM_JOB_ID