
source ~/.bashrc
conda activate LLM
cd /home/as14229/NYU_HPC/LLM-Personality-Codebase

# jobs
python template/selection.py  --version facebook/opt-6.7b  --config 'config/template-selection/index/OPT.yaml' --verbose
python template/selection.py  --version facebook/opt-2.7b  --config 'config/template-selection/index/OPT.yaml' --verbose


python template/selection.py  --version facebook/opt-6.7b  --config 'config/template-selection/non-index/OPT.yaml' --verbose
python template/selection.py  --version facebook/opt-2.7b  --config 'config/template-selection/non-index/OPT.yaml' --verbose