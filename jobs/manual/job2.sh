#!/bin/bash

# env
source ~/.bashrc
conda activate LLM
cd /home/as14229/NYU_HPC/LLM-Personality-Codebase


# jobs
python3 run.py --config 'OPT-13B' --order 'order-I' --indexed --ans index 
python3 run.py --config 'OPT-6.7B' --order 'original' --indexed --ans index 
python3 run.py --config 'OPT-6.7B' --order 'order-II' --indexed --ans index 
python3 run.py --config 'GPT2' --indexed --ans index
python3 run.py --config 'GPT2-XL' --indexed --ans index



scancel $SLURM_JOB_ID