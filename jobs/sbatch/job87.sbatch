#!/bin/bash   
#SBATCH --nodes=1               
#SBATCH --ntasks-per-node=1     
#SBATCH --cpus-per-task=16      
#SBATCH --mem=128GB             
#SBATCH --time=01:00:00         
#SBATCH --job-name=Job87
#SBATCH --output=/home/as14229/NYU_HPC/LLM-Personality-Codebase/jobs/logs/batch_1/job87.log

source ~/.bashrc
conda activate LLM
cd /home/as14229/NYU_HPC/LLM-Personality-Codebase/

python3 run.py --config 'GPTNEOX' --indexed --ans index -f 600 -t 700 --order order-II