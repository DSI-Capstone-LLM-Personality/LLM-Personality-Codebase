
source ~/.bashrc
conda activate LLM
cd /home/as14229/NYU_HPC/LLM-Personality-Codebase

# jobs
python template/calibration.py  --mode SV --family OPT --version facebook/opt-6.7b  --config 'config/Constraint/calibration/non-index.yaml' 
python template/calibration.py  --mode SV --family OPT --version facebook/opt-2.7b  --config 'config/Constraint/calibration/non-index.yaml' 


python template/calibration.py  --mode SV --family OPT --version facebook/opt-6.7b  --config 'config/Constraint/calibration/index.yaml' 
python template/calibration.py  --mode SV --family OPT --version facebook/opt-2.7b  --config 'config/Constraint/calibration/index.yaml' 