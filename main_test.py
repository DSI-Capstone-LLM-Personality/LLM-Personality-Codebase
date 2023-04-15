import os
import argparse
import yaml
from MPI.mpi import *
from model.language_model import *
from template.template import *
from util.utils import *

print(colored.fg("#ffbf00") + Style.BRIGHT + line(n=120, is_print=False))
if torch.backends.mps.is_available():
    print("-- MPS is built: ", torch.backends.mps.is_built())
    print("-- Let's use GPUs!")
elif torch.cuda.is_available():
    print(f"-- Current Device: {torch.cuda.get_device_name(0)}")
    print(
        f"-- Device Total Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print("-- Let's use", torch.cuda.device_count(), "GPUs!")
else:
    print("-- Unfortunately, we are only using CPUs now.")
line(n=120)


# Test: BART LARGE ZERO-SHOT CLASSIFIER

