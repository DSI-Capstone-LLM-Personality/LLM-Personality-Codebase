#!/bin/bash

# env
source ~/.bashrc
conda activate LLM
cd /home/as14229/NYU_HPC/LLM-Personality-Codebase


# jobs
python3 run.py --config 'OPT-13B' --order 'original'
python3 run.py --config 'OPT-13B' --order 'order-III'
python3 run.py --config 'OPT-13B' --order 'reverse'
python3 run.py --config 'OPT-13B' --order 'order-I'
python3 run.py --config 'OPT-13B' --order 'order-II'


scancel $SLURM_JOB_ID

# srun -c 4 --mem=128GB --gres=gpu:a100:1 --time=4:00:00 -J A100S --mail-user=as14229@nyu.edu --mail-type=BEGIN -v --pty /bin/bash -c "./scripts/hostname.sh; bash --login;bash /home/as14229/NYU_HPC/LLM-Personality-Codebase/jobs/job1.sh"
