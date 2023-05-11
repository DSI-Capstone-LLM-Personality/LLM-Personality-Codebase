#!/bin/bash

# env
source ~/.bashrc
conda activate LLM
cd /home/as14229/NYU_HPC/LLM-Personality-Codebase


# jobs
python3 run.py --config 'OPT-13B' --order 'reverse' --indexed
python3 run.py --config 'OPT-13B' --order 'order-I' --indexed
python3 run.py --config 'OPT-13B' --order 'order-II' --indexed
# python3 run.py --config 'OPT-6.7B' --order 'original'
# python3 run.py --config 'GPT2-Medium'
# python3 run.py --config 'GPT2-Large'
# python3 run.py --config 'GPTNEO-125M'

scancel $SLURM_JOB_ID