#!/bin/bash

# env
source ~/.bashrc
conda activate LLM
cd /home/as14229/NYU_HPC/LLM-Personality-Codebase


# jobs
python3 run.py --config 'OPT-13B' --order 'order-III' --indexed --ans index 
python3 run.py --config 'OPT-6.7B' --order 'order-I' --indexed --ans index 
python3 run.py --config 'OPT-2.7B' --indexed --ans index
python3 run.py --config 'GPTNEO-2.7B' --indexed --ans index
python3 run.py --config 'GPTNEO-1.3B' --indexed --ans index


scancel $SLURM_JOB_ID