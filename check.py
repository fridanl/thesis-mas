import pandas as pd
from pathlib import Path

direc = "/home/rp-fril-mhpe/subsampled_input_round2"
direc = Path(direc)
files = list(direc.glob(f"*.csv"))

for i, file in enumerate(files):
    print(f"looking at file {i}: {file}")
    df = pd.read_csv(file)

    print(f"this is the length of the df: {df.shape}")
    grouped = df.groupby("match_type")["id"].size().reset_index()
    print(grouped)
