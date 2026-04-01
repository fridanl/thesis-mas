import pandas as pd 
import argparse
import pathlib, yaml
from pathlib import Path 
import random

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


def load_all_as_dataframe(df_dict) -> pd.DataFrame:
    """Load all first round results and tag with model column."""
    dfs = []
    for model, df in df_dict.items():
        print(f'looking at model {model}')
        dfs.append(df)
    
    #DEBUG: 
    print(f'number of dataframes: {len(dfs)}')
    return pd.concat(dfs, ignore_index=True)

def load_first_round_results(base: Path, models: list, dataset: str, failed: bool = False) -> dict[str, pd.DataFrame]:
    '''
    Load all model results for a dataset from first round. 
    failed: indicates whether to load all failed examples instead. 
    '''

    suffix = '-failed' if failed else ''
    results = {}
    for model in models:
        print(f'model: {model}')
        path = base / f'{model}-{dataset}{suffix}.csv'
        print(f'path: {path}')
        if path.exists():
            results[model] = pd.read_csv(path)

    return results 

def load_second_round_input(base: Path, models: list, dataset: str, agreeing: bool = False) -> dict[str, pd.DataFrame]:
    '''
    Load all second round input files. 
    '''
    results = {}
    type_agreement = 'agree' if agreeing else 'disagree'
    for model in models:
        path = base / 'input_round2' / dataset / f'{model}_{type_agreement}.csv' 
        if path.exists():
            results[model] = pd.read_csv(path)
    return results

def load_second_round_subsampled(base: Path, models: list, dataset: str, agreeing: bool = False) -> dict[str, pd.DataFrame]:
    '''
    Load subsampled input for second round. 
    '''

    type_agreement = 'agree' if agreeing else 'disagree'
    results = {}
    for model in models: 
        path = base / 'subsampled_input_round2' / dataset / f'{model}_{type_agreement}_subsampled.csv'
        if path.exists():
            results[model] = pd.read_csv(path)
    return results

def print_first_round_stats(models, first_round, second_round_input, second_round_input_subsampled, second_round_input_agree, second_round_input_subsampled_agree, random_check=False):
    '''
    Print an overview of the first round, as well as the prepared dataset for the second round input.

    if random_check is set to True, a random check of the second round input is run and printed
    '''
    for model in models:
        print(f'############################# {model} #################################')
        print('FIRST ROUND')
        print(f'Number of rows in df: {first_round[model].shape[0]}')
        grouped = first_round[model].groupby('id').size().reset_index(name='count')
        print(f'Number of claims in df: {grouped.shape[0]}')
        print(f'Number of claims after discarding claims with failed: {grouped[grouped['count'] == 10].shape[0]}')


        second = second_round_input[model]
        second_grouped = second.groupby(['model_sender', 'match_type']).size().reset_index(name='count')
        print('Second')
        print(second_grouped)

        second_sub = second_round_input_subsampled[model]
        second_sub_grouped = second_sub.groupby(['model_sender', 'match_type']).size().reset_index(name='count')
        print('Subsampled')
        print(second_sub_grouped)

        print('########## Agreeing ##############')
        second = second_round_input_agree[model]
        second_grouped = second.groupby(['model_sender', 'match_type']).size().reset_index(name='count')
        print('Second')
        print(second_grouped)

        second_sub = second_round_input_subsampled_agree[model]
        second_sub_grouped = second_sub.groupby(['model_sender', 'match_type']).size().reset_index(name='count')
        print('Subsampled')
        print(second_sub_grouped)

        if random_check:
            print('SECOND ROUND')
            second_round = second_round_input[model]
            for sender in list(second_round['model_sender'].unique()):
                if model == sender or sender not in models:
                    continue
                print(f'SENDER: {sender}')
                first_round_sender = first_round[sender]
                first_round_receiver = first_round[model]
                second_round_pair = second_round[second_round['model_sender'] == sender]

                print('Looking at 3 random samples')
                candidates = list(second_round['id'].unique())
                rands = random.sample(candidates, k=3)
                for i in rands: 
                    print('First round results for receiver:\n')
                    print(first_round_receiver[first_round_receiver['id'] == i][['id', 'label']])
                    print('First round for sender:\n')
                    print(first_round_sender[first_round_sender['id'] == i][['id', 'label']])
                    print(f'From the second round input for model: {model} as receiver')

                    print(second_round_pair[second_round_pair['id'] == i][['id', 'model_receiver', 'model_sender','label_receiver', 'label_sender','match_type']])



def make_sender_receiver_matrix(inputs, model_names, model_pairs):
    '''
    prints LaTeX tables in terminal, with a count of each of the lines of input for each of the cases
    for all model pairs
    '''
    labels = ['0', '1', 'B']
    seen_pairs = set() # only print every model pair, but not both acting as sender/receicer, but only looking at that combination

    for receiver in model_names:
        print("\n" + "="*80)
        print(f"% RECEIVER: {receiver}")
        print("="*80)
        print(inputs.columns)
        slice_df = inputs[inputs['model_receiver'] == receiver]

        for sender in model_pairs.get(receiver, []):
            pair_key = frozenset([receiver, sender])
            
            # skip if we already looked at this pair
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            sender_slice = slice_df[slice_df['model_sender'] == sender]

            # Count match_types
            counts = sender_slice['match_type'].value_counts().to_dict()

            # Initialize 3x3 table
            table = {s: {r: 0 for r in labels} for s in labels}

            # Fill table
            for match, count in counts.items():
                try:
                    r_label, s_label = match.split(":")
                    if r_label in labels and s_label in labels:
                        table[s_label][r_label] = count
                except:
                    continue

            # make latex
            print(f"\n% MODEL PAIR: {receiver} -- {sender}")
            print("\\begin{table}[h!]")
            print("\\centering")
            print("\\begin{tabular}{c|ccc}")
            print("\\toprule")
            print(f"\\multirow{{2}}{{*}}{{\\textbf{{{sender}}}}} & \\multicolumn{{3}}{{c}}{{\\textbf{{{receiver}}}}} \\\\")
            print('\\cmidrule(l){2-4}')
            print(" & 0 & 1 & B \\\\")
            print("\\midrule")

            for s in labels:
                row = " & ".join(str(table[s][r]) for r in labels)
                print(f"{s} & {row} \\\\")

            print("\\bottomrule")
            print("\\end{tabular}")
            print("\\end{table}")

def main(args):
    base = Path(args.base_path)
    profiles_root = yaml.safe_load(Path('configs/models.yaml').read_text())
    profiles = profiles_root.get('profiles', {})
    models = list(profiles.keys())

    model_pairs = {"llama-3.3-70b": ["llama-3.1-8b", "qwen-2.5-72b", "gemma-3-27b", "gpt-oss-20b"], # matching with big models and family
                   "llama-3.1-8b": ["llama-3.3-70b"],
                   "qwen-2.5-72b": ["qwen-2.5-7b", "llama-3.3-70b", "gemma-3-27b", "gpt-oss-20b"], # only matching with same family
                   "qwen-2.5-7b": ["qwen-2.5-72b"],
                   "gemma-3-27b": ["gemma-3-4b", "llama-3.3-70b", "qwen-2.5-72b", "gpt-oss-20b"],
                   "gemma-3-4b": ["gemma-3-27b"],
                   "gpt-oss-20b": ["llama-3.3-70b", "qwen-2.5-72b", "gemma-3-27b"]}

    print('Running: second_round_input')
    
    second_round_input = load_second_round_input(base, models, args.dataset, agreeing=False)
    print(f'Running load_all_as_dataframe(second_round_input)')
    inputs = load_all_as_dataframe(second_round_input)

    make_sender_receiver_matrix(inputs, model_names=models, model_pairs=model_pairs)

    ## first_round = load_first_round_results(base, models, args.dataset, failed=False)
    # second_round_input_subsampled = load_second_round_subsampled(base, models, args.dataset, agreeing=False)
    # second_round_input_agree = load_second_round_input(base, models, args.dataset, agreeing=True)
    # second_round_input_subsampled_agree = load_second_round_subsampled(base, models, args.dataset, agreeing=True)
    
    ## printing the first round input stats here:
    # print_first_round_stats(models, first_round, second_round_input, second_round_input_subsampled, second_round_input_agree, second_round_input_subsampled_agree, random_check=False)
    

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_path',
                    default='/home/rp-fril-mhpe',
                    help='base path for results.')
    ap.add_argument('--dataset',
                    help='Specify name of dataset.')
    args = ap.parse_args()
    main(args)











