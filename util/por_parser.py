import pandas as pd
import numpy as np
import pyreadstat
import os
import time

# Human data parser

start = time.time()
df, meta = pyreadstat.read_por("Dataset/IPIP120.por")
print(df.head())
df.to_csv("processed.csv")
stop = time.time()
print(
    f"Processing Time: {np.round(stop - start, 2)}s | About {np.round((stop-start)/60, 1)} mins")
