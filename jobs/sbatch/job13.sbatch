#!/bin/bash   
#SBATCH --nodes=1               
#SBATCH --ntasks-per-node=1     
#SBATCH --cpus-per-task=32      
#SBATCH --mem=256GB             
#SBATCH --time=4:00:00         
#SBATCH --job-name=Job13
#SBATCH --output=/home/as14229/NYU_HPC/LLM-Personality-Codebase/jobs/logs/batch_4/job13.log

export SSH_TTY=$(tty)
export H_NAME=$(hostname)

ssh log1 ssh $H_NAME <<EOF

source ~/.bashrc
conda activate LLM
cd /home/as14229/NYU_HPC/LLM-Personality-Codebase/

python template/selection.py --verbose --config 'config/template-selection/non-index/OPT-66B.yaml' --template [og]-[ns]-[type-ii]-[ans-i]

EOF