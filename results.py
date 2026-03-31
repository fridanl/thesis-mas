import pandas as pd
from pathlib import Path 
import argparse
import yaml 
import random


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


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
    models.remove('gpt-oss-9b')

    first_round = load_first_round_results(base, models, args.dataset, failed = False)
    second_round_input = load_second_round_input(base, models, args.dataset, agreeing=False)
    second_round_input_subsampled = load_second_round_subsampled(base, models, args.dataset, agreeing=False)
    second_round_input_agree = load_second_round_input(base, models, args.dataset, agreeing=True)
    second_round_input_subsampled_agree = load_second_round_subsampled(base, models, args.dataset, agreeing=True)

    # First we check first run. 
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


        # CODE FOR RANDOM CHECK OF CONSTRUCTED INPUT TO SECOND ROUND. 
        # print('SECOND ROUND')
        # second_round = second_round_input[model]
        # for sender in list(second_round['model_sender'].unique()):
        #     if model == sender or sender not in models:
        #         continue
        #     print(f'SENDER: {sender}')
        #     first_round_sender = first_round[sender]
        #     first_round_receiver = first_round[model]
        #     second_round_pair = second_round[second_round['model_sender'] == sender]

        #     print('Looking at 3 random samples')
        #     candidates = list(second_round['id'].unique())
        #     rands = random.sample(candidates, k=3)
        #     for i in rands: 
        #         print('First round results for receiver:\n')
        #         print(first_round_receiver[first_round_receiver['id'] == i][['id', 'label']])
        #         print('First round for sender:\n')
        #         print(first_round_sender[first_round_sender['id'] == i][['id', 'label']])
        #         print(f'From the second round input for model: {model} as receiver')

        #         print(second_round_pair[second_round_pair['id'] == i][['id', 'model_receiver', 'model_sender','label_receiver', 'label_sender','match_type']])

    


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_path',
                    default='/home/rp-fril-mhpe',
                    help='base path for results.')
    ap.add_argument('--dataset',
                    help='Specify name of dataset.')
    args = ap.parse_args()
    main(args)
