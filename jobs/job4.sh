#!/bin/bash

# env
source ~/.bashrc
conda activate LLM
cd /home/as14229/NYU_HPC/LLM-Personality-Codebase


# jobs
python3 run.py --config OPT-13B --order 'order-II'
python3 run.py --config 'OPT-6.7B' --order 'order-I'
python3 run.py --config 'OPT-2.7B'
python3 run.py --config 'GPT2-XL'
python3 run.py --config 'OPT-350M'


scancel $SLURM_JOB_ID