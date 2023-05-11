#!/bin/bash   
#SBATCH --nodes=1               
#SBATCH --ntasks-per-node=1     
#SBATCH --cpus-per-task=32      
#SBATCH --mem=256GB             
#SBATCH --time=01:15:00         
#SBATCH --job-name=Job11
#SBATCH --output=/home/as14229/NYU_HPC/LLM-Personality-Codebase/jobs/logs/batch_2/job11.log

source ~/.bashrc
conda activate LLM
cd /home/as14229/NYU_HPC/LLM-Personality-Codebase/

python template/selection.py --verbose --config 'config/template-selection/OPT-66B.yaml' --temp_idx 10