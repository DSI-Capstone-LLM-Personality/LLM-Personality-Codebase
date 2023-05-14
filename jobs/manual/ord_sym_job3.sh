#!/bin/bash

# env
source ~/.bashrc
conda activate LLM
cd /home/as14229/NYU_HPC/LLM-Personality-Codebase


# jobs
python3 run.py --config 'OPT-13B' --order 'order-II'
python3 run.py --config 'OPT-6.7B' --order 'reverse'
python3 run.py --config 'OPT-6.7B' --order 'order-III'
python3 run.py --config 'OPT-1.3B'
python3 run.py --config 'OPT-350M'
python3 run.py --config 'OPT-125M'
python3 run.py --config 'GPTNEO-1.3B'


python3 run.py --config 'OPT-13B' --order 'order-II' --indexed --ans index
python3 run.py --config 'OPT-6.7B' --order 'reverse' --indexed --ans index
python3 run.py --config 'OPT-6.7B' --order 'order-III' --indexed --ans index
python3 run.py --config 'OPT-1.3B' --indexed --ans index
python3 run.py --config 'OPT-350M' --indexed --ans index
python3 run.py --config 'OPT-125M' --indexed --ans index


# scancel $SLURM_JOB_ID