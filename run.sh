#!/bin/bash
# whoami
# date
COLOR="\e[31m"
ENDCOLOR="\e[0m"
LINE=$(printf "%60s")
echo -e $(printf "${COLOR}${LINE// /=}${ENDCOLOR}")
echo -e $(printf "${COLOR}JOB STARTED!!!${ENDCOLOR}")
# DO NOT MODIFY ABOVE

# TO RUN: change the following as needed:
# REGIME="Constraint"
REGIME="Open-Vocab" # uncomment this if necessary
TYPE="order-symmetry"
MODEL="BART-Large" #
driver="non-index.yaml"

# Declare file path & Driver files
FILE="config/${REGIME}/${TYPE}/${MODEL}/${driver}"
script="main_mpi.py"
# echo $FILE
declare -a ORDERS=("original" "reverse" "order-I" "order-II" "order-III")
for ((i=0; i<${#ORDERS[@]}; i++)); do
    echo -e $(printf "${COLOR}${LINE// /=}${ENDCOLOR}")
    echo -e $(printf "${COLOR}CURRENT RUNNING ORDER: ${ORDERS[i]}${ENDCOLOR}")
    echo -e $(printf "${COLOR}${LINE// /=}${ENDCOLOR}")
    yaml-set $FILE --change='/shuffle/order' --value=${ORDERS[i]}
    python3 $script --config=$FILE --verbose
done

echo -e $(printf "${COLOR}${LINE// /=}${ENDCOLOR}")
echo -e $(printf "${COLOR}JOB FINISHED!!!${ENDCOLOR}")
echo -e $(printf "${COLOR}${LINE// /=}${ENDCOLOR}")