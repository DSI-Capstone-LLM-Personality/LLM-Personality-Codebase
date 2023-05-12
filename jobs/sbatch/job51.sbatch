#!/bin/bash   
#SBATCH --nodes=1               
#SBATCH --ntasks-per-node=1     
#SBATCH --cpus-per-task=16      
#SBATCH --mem=128GB             
#SBATCH --time=01:15:00         
#SBATCH --job-name=Job50
#SBATCH --output=/home/as14229/NYU_HPC/LLM-Personality-Codebase/jobs/logs/batch_1/job50.log

hname=$(hostname)

# ssh -T log1 ssh -T $hname <<EOF
# source ~/.bashrc
# conda activate LLM
# cd /home/as14229/NYU_HPC/LLM-Personality-Codebase/

# python3 run.py --config 'OPT-66B' --ans index  -f 0 -t 50 --order order-III

# EOF

ssh log1 ssh $hname "\
echo $(hostname)
source ~/.bashrc;\
conda activate LLM;\
cd /home/as14229/NYU_HPC/LLM-Personality-Codebase/;\
\
python3 run.py --config 'OPT-66B' --ans index  -f 0 -t 50 --order order-III;"