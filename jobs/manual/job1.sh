#!/bin/bash

# env
source ~/.bashrc
conda activate LLM
cd /home/as14229/NYU_HPC/LLM-Personality-Codebase


# jobs
python3 run.py --config 'OPT-13B' --order 'original' --indexed --ans index 
python3 run.py --config 'OPT-13B' --order 'reverse' --indexed --ans index 
python3 run.py --config 'GPT2-Medium' --indexed --ans index
python3 run.py --config 'GPT2-Large' --indexed --ans index
python3 run.py --config 'GPTNEO-125M' --indexed --ans index


scancel $SLURM_JOB_ID

# srun -c 16 --mem=256GB --gres=gpu:mi250:1 --time=4:00:00 -J A100S --mail-user=as14229@nyu.edu --mail-type=BEGIN -v --pty /bin/bash -c "./scripts/hostname.sh; bash --login;bash /home/as14229/NYU_HPC/LLM-Personality-Codebase/jobs/job1.sh"
