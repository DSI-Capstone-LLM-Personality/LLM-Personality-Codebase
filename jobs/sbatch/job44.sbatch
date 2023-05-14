#!/bin/bash   
#SBATCH --nodes=1               
#SBATCH --ntasks-per-node=1     
#SBATCH --cpus-per-task=16      
#SBATCH --mem=128GB             
#SBATCH --time=01:00:00         
#SBATCH --job-name=Job44
#SBATCH --output=/home/as14229/NYU_HPC/LLM-Personality-Codebase/jobs/logs/batch_1/job44.log

source ~/.bashrc
conda activate LLM
cd /home/as14229/NYU_HPC/LLM-Personality-Codebase/

python3 run.py --config 'GPTNEOX' -f 300 -t 400 --order order-III