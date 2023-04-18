import pandas as pd
import numpy as np
import pyreadstat
import os
import time

# Human data parser


def process_raw_por(filename):
    start = time.time()
    df, meta = pyreadstat.read_por(f"Dataset/Human Data/{filename}.por")
    print(df.head())
    df.to_csv(f"Dataset/Human Data/{filename}.csv")
    stop = time.time()
    print(
        f"Processing Time: {np.round(stop - start, 2)}s | About {np.round((stop-start)/60, 1)} mins")

# process_raw_por("IPIP120")
# process_raw_por("IPIP300")
