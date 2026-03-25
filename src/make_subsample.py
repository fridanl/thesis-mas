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
    max_ids = cap // 10 # since this is done at a claim level
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
                    help = "suffix of the files to procces (either disagree or agree)")
    ap.add_argument("--input_dir",
                    default="/home/rp-fril-mhpe/input_round2")
    ap.add_argument("--output_dir",
                    default="/home/rp-fril-mhpe/subsampled_input_round2")
    args = ap.parse_args()
    main(args)



# import pandas as pd
# import numpy as np
# import argparse
# from pathlib import Path
# from collections import defaultdict

# CAP = 2 # set this to 50_000 

# def get_effective_match_type(row):
#     """
#     To ensure that we get a balanced selection of B:B,
#     for both directions, since this is not excplicit in the mat_type column
#     """
#     if row["match_type"] != "B:B":
#         return row["match_type"]

#     # derive direction from labels
#     if row["label_sender"] == 0 and row["label_receiver"] == 1:
#         return "0:1_from_BB"
#     elif row["label_sender"] == 1 and row["label_receiver"] == 0:
#         return "1:0_from_BB"
#     else:
#         return "other_BB"  # this will not happen, but just in case


# def subsample(path, out_path, cap=CAP):
#     print(f'\nProcessing {path.name}...')
#     max_ids = cap // 10 # since this is done at a claim level
#     df = pd.read_csv(path)

#     print(f'the shape of the df: {df.shape}')

#     print("applying the B:B match type function...")
#     df["effective_match_type"] = df.apply(get_effective_match_type, axis=1)
#     print(f'the shape of the df: {df.shape}')

#     # compute rows per id
#     id_to_type = df.groupby("id")["effective_match_type"].unique()
#     print(f'id_to_type: {id_to_type}')

#     type_to_ids = defaultdict(list)
#     for claim_id, types in id_to_type.items():
#         for t in types:
#             type_to_ids[t].append(claim_id)

#     selected_ids = set()

#     for match_type, ids in type_to_ids.items():
#         ids = np.array(ids)
#         print(f'{match_type}: {len(ids)} ids')

#         if len(ids) <= max_ids: # then just keep those we have
#             print(f'the length of ids {len(ids)} is smaller than the max_ids {max_ids} ')
#             chosen = ids
#             print(f'keeping all {len(ids)} ids, since its smaller than the cap of {cap}')

#         else: # shuffle and select a subset
#             np.random.shuffle(ids)
#             chosen = ids[:max_ids]


#         selected_ids.update(chosen)

#         print(f"{match_type}: selected {len(chosen)} ids")

#     # filter df to selected ids
#     df_sub = df[df["id"].map(selected_ids.__contains__)]

#     df_sub.to_csv(out_path, index=False)

#     print(f"-> saved to {out_path.name} ({len(df_sub)} rows)")


# def process_folder(input_dir, output_dir, suffix):
#     input_dir = Path(input_dir)
#     output_dir = Path(output_dir)

#     output_dir.mkdir(parents=True, exist_ok=True)

#     files =list(input_dir.glob(f'*_{suffix}.csv'))
#     print(f'found {len(files)} files')

#     for i, file in enumerate(files):
#         print(f'\n[{i}/{len(files)}]')
#         out_file = output_dir / f'{file.stem}_subsampled.csv'
#         subsample(file, out_file)


# def main(args):
#     process_folder(args.input_dir, args.output_dir, suffix=args.suffix)

# if __name__ == "__main__":
#     ap = argparse.ArgumentParser(description="subsample the input for round 2, to reduce the size for experiment")
#     ap.add_argument("--suffix", 
#                     type=str,
#                     default="disagree",
#                     help = "suffix of the files to procces (either disagree or agree)")
#     ap.add_argument("--input_dir",
#                     default="/home/rp-fril-mhpe/input_round2")
#     ap.add_argument("--output_dir",
#                     default="/home/rp-fril-mhpe/subsampled_input_round2")
#     args = ap.parse_args()
#     main(args)