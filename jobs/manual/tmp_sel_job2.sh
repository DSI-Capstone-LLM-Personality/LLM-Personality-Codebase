
source ~/.bashrc
conda activate LLM
cd /home/as14229/NYU_HPC/LLM-Personality-Codebase

# jobs
python template/selection.py  --version facebook/opt-13b   --config 'config/template-selection/index/OPT.yaml' --verbose
python template/selection.py  --version EleutherAI/gpt-neo-125m  --config 'config/template-selection/index/GPTNEO.yaml' --verbose


python template/selection.py  --version facebook/opt-13b   --config 'config/template-selection/non-index/OPT.yaml' --verbose
python template/selection.py  --version EleutherAI/gpt-neo-125m  --config 'config/template-selection/non-index/GPTNEO.yaml' --verbose