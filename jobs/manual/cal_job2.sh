
source ~/.bashrc
conda activate LLM
cd /home/as14229/NYU_HPC/LLM-Personality-Codebase

# jobs
python template/calibration.py  --mode SV --family OPT --version facebook/opt-13b  --config 'config/Constraint/calibration/non-index.yaml' 
python template/calibration.py  --mode SV --family GPTNEO --version EleutherAI/gpt-neo-125m  --config 'config/Constraint/calibration/non-index.yaml' 


python template/calibration.py  --mode SV --family OPT --version facebook/opt-13b   --config 'config/Constraint/calibration/index.yaml' 
python template/calibration.py  --mode SV --family GPTNEO --version EleutherAI/gpt-neo-125m  --config 'config/Constraint/calibration/index.yaml' 