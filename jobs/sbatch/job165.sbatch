#!/bin/bash   
#SBATCH --nodes=1               
#SBATCH --ntasks-per-node=1     
#SBATCH --cpus-per-task=16      
#SBATCH --mem=128GB             
#SBATCH --time=01:30:00         
#SBATCH --job-name=Job165
#SBATCH --output=/home/as14229/NYU_HPC/LLM-Personality-Codebase/jobs/logs/batch_3/job165.log

source ~/.bashrc
conda activate LLM
cd /home/as14229/NYU_HPC/LLM-Personality-Codebase/

python3 run.py --config 'OPT-30B' -f 0 -t 100 --order reverse