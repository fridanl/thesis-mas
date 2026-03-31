import pandas as pd
from pathlib import Path 
import argparse
import yaml 


def load_all_as_dataframe(df_dict) -> pd.DataFrame:
    """Load all first round results and tag with model column."""
    dfs = []
    for model, df in df_dict.items():
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def load_first_round_results(base: Path, models: list, dataset: str, failed: bool = False) -> dict[str, pd.DataFrame]:
    '''
    Load all model results for a dataset from first round. 
    failed: indicates whether to load all failed examples instead. 
    '''

    suffix = '-failed' if failed else ''
    results = {}
    for model in models:
        path = base / f'{model}-{dataset}{suffix}.csv'
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



def main(args):
    base = Path(args.base_path)
    profiles_root = yaml.safe_load(Path('configs/models.yaml').read_text())
    profiles = profiles_root.get('profiles', {})
    models = list(profiles.keys())
    models = ['gemma-3-27b']

    first_round = load_first_round_results(base, models, 'sarcasm', failed = False)
    second_round_input = load_second_round_input(base, models, 'sarcasm', agreeing=False)
    second_round_input_subsampled = load_second_round_subsampled(base, models, 'sarcasm', agreeing=False)

    for model in models:
        print('This is the results df')
        print(first_round[model].head())
        print(first_round[model].shape)

        print('this is the second round input')
        print(second_round_input[model].head())
        print(second_round_input[model].shape)

        print('this is the second round input subsampled')
        print(second_round_input_subsampled[model].head())
        print(second_round_input_subsampled[model].shape)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_path',
                    default='/home/rp-fril-mhpe',
                    help='base path for results.')
    ap.add_argument('--dataset',
                    help='Specify name of dataset.')
    args = ap.parse_args()
    main(args)