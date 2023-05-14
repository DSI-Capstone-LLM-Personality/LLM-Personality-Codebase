#!/bin/bash

# OPT - index
# python human/stats.py --config=config/Constraint/order-symmetry/OPT-125M/index.yaml
# python human/stats.py --config=config/Constraint/order-symmetry/OPT-350M/index.yaml
# python human/stats.py --config=config/Constraint/order-symmetry/OPT-1.3B/index.yaml
# python human/stats.py --config=config/Constraint/order-symmetry/OPT-2.7B/index.yaml
# python human/stats.py --config=config/Constraint/order-symmetry/OPT-6.7B/index.yaml
# python human/stats.py --config=config/Constraint/order-symmetry/OPT-13B/index.yaml


# OPT - non-index
python human/stats.py --config=config/Constraint/order-symmetry/OPT-125M/non-index.yaml
python human/stats.py --config=config/Constraint/order-symmetry/OPT-350M/non-index.yaml
python human/stats.py --config=config/Constraint/order-symmetry/OPT-1.3B/non-index.yaml
python human/stats.py --config=config/Constraint/order-symmetry/OPT-2.7B/non-index.yaml
python human/stats.py --config=config/Constraint/order-symmetry/OPT-6.7B/non-index.yaml
python human/stats.py --config=config/Constraint/order-symmetry/OPT-13B/non-index.yaml
