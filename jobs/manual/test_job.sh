#!/bin/bash

# Set SSH_TTY environment variable
export SSH_TTY=$(tty)
export H_NAME=$(hostname)

# ssh to log1 and execute multiple commands on cm041
ssh log1 ssh $H_NAME <<EOF

source ~/.bashrc
conda activate LLM
cd /home/as14229/NYU_HPC/LLM-Personality-Codebase
python template/selection.py --verbose --config 'config/template-selection/index/OPT-66B.yaml' --temp_idx 0

EOF