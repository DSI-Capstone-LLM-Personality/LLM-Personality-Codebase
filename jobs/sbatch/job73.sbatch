#!/bin/bash   
#SBATCH --nodes=1               
#SBATCH --ntasks-per-node=1     
#SBATCH --cpus-per-task=16      
#SBATCH --mem=256GB             
#SBATCH --time=01:00:00         
#SBATCH --job-name=Job73
#SBATCH --output=/home/as14229/NYU_HPC/LLM-Personality-Codebase/jobs/logs/batch_6/job73.log

export SSH_TTY=$(tty)
export H_NAME=$(hostname)

ssh log1 ssh $H_NAME <<EOF

source ~/.bashrc
conda activate LLM
cd /home/as14229/NYU_HPC/LLM-Personality-Codebase/

python3 run.py --config 'OPT-66B' -f 600 -t 650 --order reverse

EOF