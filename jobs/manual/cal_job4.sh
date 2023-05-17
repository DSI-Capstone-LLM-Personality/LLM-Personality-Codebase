
source ~/.bashrc
conda activate LLM
cd /home/as14229/NYU_HPC/LLM-Personality-Codebase

# jobs
python template/calibration.py --version EleutherAI/gpt-neo-2.7b  --mode SV --family GPTNEO --config 'config/Constraint/calibration/non-index.yaml' 
python template/calibration.py --version EleutherAI/gpt-neo-1.3b  --mode SV --family GPTNEO --config 'config/Constraint/calibration/non-index.yaml' 
python template/calibration.py --version facebook/opt-1.3b        --mode SV --family OPT    --config 'config/Constraint/calibration/non-index.yaml' 
python template/calibration.py --version facebook/opt-125m        --mode SV --family OPT    --config 'config/Constraint/calibration/non-index.yaml' 
python template/calibration.py --version facebook/opt-350m        --mode SV --family OPT    --config 'config/Constraint/calibration/non-index.yaml' 


python template/calibration.py --version EleutherAI/gpt-neo-2.7b  --mode SV --family GPTNEO --config 'config/Constraint/calibration/index.yaml' 
python template/calibration.py --version EleutherAI/gpt-neo-1.3b  --mode SV --family GPTNEO --config 'config/Constraint/calibration/index.yaml' 
python template/calibration.py --version facebook/opt-1.3b        --mode SV --family OPT    --config 'config/Constraint/calibration/index.yaml' 
python template/calibration.py --version facebook/opt-125m        --mode SV --family OPT    --config 'config/Constraint/calibration/index.yaml' 
python template/calibration.py --version facebook/opt-350m        --mode SV --family OPT    --config 'config/Constraint/calibration/index.yaml' 