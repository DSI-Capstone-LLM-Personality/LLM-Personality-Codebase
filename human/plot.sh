#!/bin/bash

# OPT
python human/stats.py --config=config/Constraint/order-symmetry/OPT-125M/index.yaml
python human/stats.py --config=config/Constraint/order-symmetry/OPT-350M/index.yaml
python human/stats.py --config=config/Constraint/order-symmetry/OPT-1.3B/index.yaml
python human/stats.py --config=config/Constraint/order-symmetry/OPT-2.7B/index.yaml
python human/stats.py --config=config/Constraint/order-symmetry/OPT-6.7B/index.yaml
# python human/plot.py --config=config/Constraint/order-symmetry/OPT-13B/index.yaml
