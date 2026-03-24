import pandas as pd
import numpy as np
import argparse
from pathlib import Path


CAP = 2 # set this to 50_000 


def subsample(path, out_path, cap=CAP):
    print(f'\nProcessing {path.name}...')

    df = pd.read_csv(path)

    # compute rows per id
    id_sizes = df.groupby("id").size() # number of rows per claim
    id_to_type = df.groupby("id")["match_type"].first()

    selected_ids = []

    for match_type, ids in id_to_type.groupby(id_to_type):
        ids = ids.index.to_numpy()
        np.random.shuffle(ids)

        current_rows = 0
        chosen = []

        for i in ids:
            rows = id_sizes[i]

            if current_rows + rows > cap: # if limit is reached, break
                break

            chosen.append(i)
            current_rows += rows

        selected_ids.extend(chosen)

        print(f"{match_type}: selected {len(chosen)} ids, total rows: {current_rows}")

    # filter df to selected ids
    df_sub = df[df["id"].map(selected_ids.__contains__)]

    df_sub.to_csv(out_path, index=False)

    print(f"-> saved to {out_path.name} ({len(df_sub)} rows)")


def process_folder(input_dir, output_dir, suffix):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    files =list(input_dir.glob(f'*_{suffix}.csv'))
    print(f'found {len(files)} files')

    for i, file in enumerate(files):
        print(f'\n[{i}/{len(files)}]')
        out_file = output_dir / f'{file.stem}_subsampled.csv'
        subsample(file, out_file)


def main(args):
    process_folder(args.input_dir, args.output_dir, suffix=args.suffix)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="subsample the input for round 2, to reduce the size for experiment")
    ap.add_argument("--suffix", 
                    type=str,
                    default="disagree",
                    help = "suffix of the files to procces (either disagree or agree)")
    ap.add_argument("--input_dir",
                    default="/home/rp-fril-mhpe/input_round2")
    ap.add_argument("--output_dir",
                    default="/home/rp-fril-mhpe/subsampled_input_round2")
    args = ap.parse_args()
    main(args)