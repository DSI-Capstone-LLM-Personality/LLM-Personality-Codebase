
source ~/.bashrc
conda activate LLM
cd /home/as14229/NYU_HPC/LLM-Personality-Codebase

# jobs
python template/selection.py  --version EleutherAI/gpt-neo-2.7b  --config 'config/template-selection/index/GPTNEO.yaml' --verbose
python template/selection.py  --version EleutherAI/gpt-neo-1.3b  --config 'config/template-selection/index/GPTNEO.yaml' --verbose
python template/selection.py  --version facebook/opt-1.3b  --config 'config/template-selection/index/OPT.yaml' --verbose
python template/selection.py  --version facebook/opt-125m  --config 'config/template-selection/index/OPT.yaml' --verbose
python template/selection.py  --version facebook/opt-350m  --config 'config/template-selection/index/OPT.yaml' --verbose


python template/selection.py  --version EleutherAI/gpt-neo-2.7b  --config 'config/template-selection/non-index/GPTNEO.yaml' --verbose
python template/selection.py  --version EleutherAI/gpt-neo-1.3b  --config 'config/template-selection/non-index/GPTNEO.yaml' --verbose
python template/selection.py  --version facebook/opt-1.3b  --config 'config/template-selection/non-index/OPT.yaml' --verbose
python template/selection.py  --version facebook/opt-125m  --config 'config/template-selection/non-index/OPT.yaml' --verbose
python template/selection.py  --version facebook/opt-350m  --config 'config/template-selection/non-index/OPT.yaml' --verbose