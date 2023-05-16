
source ~/.bashrc
conda activate LLM
cd /home/as14229/NYU_HPC/LLM-Personality-Codebase

# jobs
python template/calibration.py  --mode SV --family GPT2 --version gpt2         --config 'config/Constraint/calibration/non-index.yaml' 
python template/calibration.py  --mode SV --family GPT2 --version gpt2-medium  --config 'config/Constraint/calibration/non-index.yaml'
python template/calibration.py  --mode SV --family GPT2 --version gpt2-large   --config 'config/Constraint/calibration/non-index.yaml'
python template/calibration.py  --mode SV --family GPT2 --version gpt2-xl      --config 'config/Constraint/calibration/non-index.yaml'


python template/calibration.py  --mode SV --family GPT2 --version gpt2         --config 'config/Constraint/calibration/index.yaml' 
python template/calibration.py  --mode SV --family GPT2 --version gpt2-medium  --config 'config/Constraint/calibration/index.yaml'
python template/calibration.py  --mode SV --family GPT2 --version gpt2-large   --config 'config/Constraint/calibration/index.yaml'
python template/calibration.py  --mode SV --family GPT2 --version gpt2-xl      --config 'config/Constraint/calibration/index.yaml'