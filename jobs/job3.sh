#!/bin/bash

# env
source ~/.bashrc
conda activate LLM
cd /home/as14229/NYU_HPC/LLM-Personality-Codebase


# jobs
# python3 run.py --config 'OPT-13B' --order 'order-I'
python3 run.py --config 'OPT-6.7B' --order 'reverse'
python3 run.py --config 'OPT-6.7B' --order 'order-III'
python3 run.py --config 'GPT2'
python3 run.py --config 'GPTNEO-2.7B'


scancel $SLURM_JOB_ID