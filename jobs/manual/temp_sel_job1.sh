
source ~/.bashrc
conda activate LLM
cd /home/as14229/NYU_HPC/LLM-Personality-Codebase

# jobs
python template/selection.py  --version gpt2         --config 'config/template-selection/index/GPT2.yaml' --verbose
python template/selection.py  --version gpt2-medium  --config 'config/template-selection/index/GPT2.yaml' --verbose
python template/selection.py  --version gpt2-large   --config 'config/template-selection/index/GPT2.yaml' --verbose
python template/selection.py  --version gpt2-xl      --config 'config/template-selection/index/GPT2.yaml' --verbose


python template/selection.py  --version gpt2         --config 'config/template-selection/non-index/GPT2.yaml' --verbose
python template/selection.py  --version gpt2-medium  --config 'config/template-selection/non-index/GPT2.yaml' --verbose
python template/selection.py  --version gpt2-large   --config 'config/template-selection/non-index/GPT2.yaml' --verbose
python template/selection.py  --version gpt2-xl      --config 'config/template-selection/non-index/GPT2.yaml' --verbose