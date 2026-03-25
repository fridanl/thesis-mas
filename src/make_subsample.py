import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from collections import defaultdict

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', None)
# vi vælger cap der er ens over modeller

# sørg for at group by id, match_type, sender_model så der bliver subsamplet lige mange for hver model


CAP = 1 # this is the number of unique ids

def subsample(path, out_path, cap=CAP):
    print(f'\nProcessing {path.name}...')
    df = pd.read_csv(path)
    
    senders = df["model_sender"].unique()
    match_types = df["match_type"].unique()

    dfs = []
    for sender in senders:
        print(f'Looking at sender: {sender}.....')
        chosen = []
        for mt in match_types:
            # cap = total 
            print(f'looking at match_type: {mt}.....')
            slic = df[(df["model_sender"] == sender) & (df["match_type"] == mt)]
            unique_ids = slic["id"].unique()
            if len(unique_ids) <= cap:
                print(f'no need to sample, appending all ids :)')
                chosen.extend(unique_ids)
                continue
            
            np.random.shuffle(unique_ids)
            print(f'Now appending {len(unique_ids[:cap])}.. \n')
            chosen.extend(unique_ids[:cap])

        chosen_df = df[df['id'].isin(chosen)]
        print(f'now appending df with {chosen_df.shape[0]}')
        dfs.append(chosen_df)

    sample_df = pd.concat(dfs, axis=0)
    print('sample df')
    print(sample_df.groupby(['model_sender', 'match_type']).size().reset_index())

    sample_df.to_csv(out_path, index=False)



def main(args):
    suffix = args.suffix
    input_dir = Path(args.input_dir)
    output_dir = Path( args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    files = list(input_dir.glob(f'*_{suffix}.csv'))
    print(f'found {len(files)} files')

    for i, file in enumerate(files):
        print(f'\n[{i}/{len(files)}]')
        out_file = output_dir / f'{file.stem}_subsampled.csv'
        print(out_file)
        subsample(file, out_file)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="subsample the input for round 2, to reduce the size for experiment")
    ap.add_argument("--suffix", 
                    type=str,
                    default="disagree",
                    help="suffix of the files to procces (either disagree or agree)")
    ap.add_argument("--cap",
                    type=int,
                    default=7000,
                    help="capacity of maximum number of ids for match_type")
    ap.add_argument("--input_dir",
                    default="/home/rp-fril-mhpe/input_round2")
    ap.add_argument("--output_dir",
                    default="/home/rp-fril-mhpe/subsampled_input_round2")
    args = ap.parse_args()
    main(args)
