import pandas as pd
import numpy as np
import argparse
from pathlib import Path

def subsample(path, out_path, cap):
    '''
    Subsampling input for second round. 
    For a given receiver (fixed in file), sample [CAP] many ids.
    '''
    print(f'\nProcessing {path.name}...')
    df = pd.read_csv(path)
    
    senders = df["model_sender"].unique()

    dfs = []
    for sender in senders:
        print(f'Looking at sender: {sender}.....')
        chosen = []
        df_sender = df[df['model_sender'] == sender] # get data specific for sender
        for mt in df_sender["match_type"].unique(): # over the match-types the pair has together
            # cap = total 
            print(f'looking at match_type: {mt}.....')
            slic = df_sender[df_sender["match_type"] == mt]
            unique_ids = slic["id"].unique() # get unique ids for (receiver, sender, match_type)
            if len(unique_ids) <= cap: # if smaller than cap, use all 
                print(f'no need to sample, appending all ids :)')
                chosen.extend(unique_ids)
                continue
            
            # take random ids at size [cap]
            np.random.shuffle(unique_ids)
            print(f'Now appending {len(unique_ids[:cap])}.. \n')
            chosen.extend(unique_ids[:cap]) # keep track of ids 

        chosen_df = df_sender[df_sender['id'].isin(chosen)] # for (receiver, sender) get data
        print(f'now appending df with {chosen_df.shape[0]}')
        dfs.append(chosen_df)

    sample_df = pd.concat(dfs, axis=0)
    print('sample df')
    print(sample_df.groupby(['model_sender', 'match_type']).size().reset_index())
    sample_df.to_csv(out_path, index=False)


def main(args):
    cap = args.cap # this is the number of unique ids
    suffix = args.suffix

    input_dir = Path(args.input_dir) / args.dataset
    output_dir = Path(args.output_dir) / args.dataset


    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = args.glob_pattern if args.glob_pattern else f'*_{suffix}.csv'
    files = list(input_dir.glob(pattern))
    
    print(f'found {len(files)} files')

    for i, file in enumerate(files):
        print(f'\n[{i}/{len(files)}]')
        out_file = output_dir / f'{file.stem}_subsampled.csv'
        print(out_file)
        subsample(file, out_file, cap=cap)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="subsample the input for round 2, to reduce the size for experiment")
    ap.add_argument("--suffix", 
                    type=str,
                    default="disagree",
                    help="suffix of the files to procces (either disagree or agree)")
    ap.add_argument("--glob_pattern",
                type=str,
                default=None,
                help="Override the default glob pattern for file matching (e.g. '*-self-interaction.csv')")
    ap.add_argument("--cap",
                    type=int,
                    default=7000,
                    help="capacity of maximum number of ids for match_type")
    ap.add_argument("--input_dir",
                    default="/home/rp-fril-mhpe/input_round2")
    ap.add_argument("--output_dir",
                    default="/home/rp-fril-mhpe/subsampled_input_round2")
    ap.add_argument('--dataset',
                    help= 'Specify the dataset of interest')
    args = ap.parse_args()
    main(args)
