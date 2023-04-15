#!/bin/bash
# whoami
# date
COLOR="\e[1;31m"
# Color themes: Yellow, Green, Cyan Blue
declare -a PROMPTCOLOR=("\e[1;33m" "\e[1;34m" "\e[1;32m" "\e[1;36m")
ENDCOLOR="\e[0m"
LINE=$(printf "%80s")
echo -e $(printf "${COLOR}${LINE// /=}${ENDCOLOR}")
echo -e $(printf "${COLOR}JOB STARTED!!!${ENDCOLOR}")
echo -e $(printf "${COLOR}${LINE// /=}${ENDCOLOR}")
# DO NOT MODIFY ABOVE

# TO RUN: change the following as needed:
# Hardcode version: deprecated
# REGIME="Constraint"
# REGIME="Open-Vocab" 
# TYPE="order-symmetry"
# MODEL="BART-Large" #
# driver="non-index.yaml"

# Read command line argument
while getopts r:t:m:d: flag
do
    case "${flag}" in
        r) REGIME=${OPTARG};;
        t) TYPE=${OPTARG};;
        m) MODEL=${OPTARG};;
        d) driver=${OPTARG};;
    esac
done
# If no command line argument is provided, prompt as needed
if [[ $# -eq 0 ]] || [[ $# -ne 8 ]] # TODO: change this later to handle more arguments
then
    echo -e $(printf "${COLOR}NO cmd line arguments given or the arguments are not valid.${ENDCOLOR}")
    echo -e $(printf "${COLOR}Please enter it manually below OR [Ctrl+C] to exit and re-enter cmd arguments.${ENDCOLOR}")
    echo -e $(printf "${COLOR}This is NOT an error. You may proceed now.${ENDCOLOR}")
    echo -e $(printf "${COLOR}${LINE// /=}${ENDCOLOR}")
    # Read-in inputs version
    echo -e $(printf "${PROMPTCOLOR[0]}${LINE// /#}${ENDCOLOR}")
    read -p "$(printf "${PROMPTCOLOR[0]}Experiment Regime/Method (\'Constraint\' or \'Open-Vocab\'): ${ENDCOLOR}")" REGIME
    echo -e $(printf "${PROMPTCOLOR[0]}${LINE// /#}${ENDCOLOR}")
    echo -e $(printf "${PROMPTCOLOR[1]}${LINE// /#}${ENDCOLOR}")
    read -p "$(printf "${PROMPTCOLOR[1]}Experiment Type (\'order-symmetry\' or \'prompt-engineering\'): ${ENDCOLOR}")" TYPE
    echo -e $(printf "${PROMPTCOLOR[1]}${LINE// /#}${ENDCOLOR}")
    echo -e $(printf "${PROMPTCOLOR[2]}${LINE// /#}${ENDCOLOR}")
    read -p "$(printf "${PROMPTCOLOR[2]}Experimet Model (please pick one from \'config\' folder): ${ENDCOLOR}")" MODEL
    echo -e $(printf "${PROMPTCOLOR[2]}${LINE// /#}${ENDCOLOR}")
    echo -e $(printf "${PROMPTCOLOR[3]}${LINE// /#}${ENDCOLOR}")
    read -p "$(printf "${PROMPTCOLOR[3]}Configuration File (please pick one from model folder, e.g. \'non-index.yaml\'): ${ENDCOLOR}")" driver
    echo -e $(printf "${PROMPTCOLOR[3]}${LINE// /#}${ENDCOLOR}")
fi

# Declare file path & Driver files
FILE="config/${REGIME}/${TYPE}/${MODEL}/${driver}"
echo -e $(printf "${COLOR}${LINE// /=}${ENDCOLOR}")
if [ -f "$FILE" ]; then
    echo -e $(printf "${COLOR}Congratulations! You PASSED Argument checking...${ENDCOLOR}")
else 
    echo -e $(printf "${COLOR}Unfortunately, your input file \'$FILE\' does not exit.${ENDCOLOR}")
    echo -e $(printf "${COLOR}Process Killed: please check your input arguments.${ENDCOLOR}")
    echo -e $(printf "${COLOR}${LINE// /=}${ENDCOLOR}")
    exit 0
fi
echo -e $(printf "${COLOR}${LINE// /=}${ENDCOLOR}")

script="main_mpi.py"
# echo $FILE
declare -a ORDERS=("original" "reverse" "order-I" "order-II" "order-III")

for ((i=0; i<${#ORDERS[@]}; i++)); do
    echo -e $(printf "${COLOR}${LINE// /=}${ENDCOLOR}")
    echo -e $(printf "${COLOR}EXPERIMENT IS RUNNING with ORDER: ${ORDERS[i]}${ENDCOLOR}")
    echo -e $(printf "${COLOR}${LINE// /=}${ENDCOLOR}")
    yaml-set $FILE --change='/shuffle/order' --value=${ORDERS[i]}
    python3 $script --config=$FILE --verbose # Use this line if you need verbose mode
#    python $script --config=$FILE # Comment this if you need verbose mode
done

echo -e $(printf "${COLOR}${LINE// /=}${ENDCOLOR}")
echo -e $(printf "${COLOR}JOB FINISHED!!!${ENDCOLOR}")
echo -e $(printf "${COLOR}${LINE// /=}${ENDCOLOR}")
