import pandas as pd 
import argparse
import pathlib, yaml
from pathlib import Path 

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

def check_results(combined, *, dataset_name, n_repetitions):
    '''Function to check if the output in files (valid and failed) correspond to expected number.'''
    datasets = {'sarcasm': 'data/sarc/sarcasm.csv'}

    if dataset_name not in datasets:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    
    path_data = Path(datasets[dataset_name])
    data = pd.read_csv(path_data, low_memory=False)

    n_claims = data.shape[0]
    expected_output_size = n_claims * n_repetitions

    print(f'[DATASET]: {dataset_name}')
    print(f'[CLAIMS IN DATASET]: {n_claims}')
    print(f'[REPETITIONS PER CLAIM]: {n_repetitions}')
    print(f'[EXPECTED ROWS PER MODEL]: {expected_output_size}')


    print(f'\n {'-'*8} PER MODEL CHECK {'-'*8}')
    print(f'[SIZE OF OUTPUT PER MODEL, GROUPED BY VALID, FAILED]:')
    output_sizes = (
        combined.groupby(['model', 'valid_json'])
        .agg(
            output_size = ('id', 'size')
        ).reset_index()
    )
    print(output_sizes)

    grouped = (
        combined.groupby(['model', 'id'])
        .agg(
            total_outputs=('id', 'size')
            ,valid_outputs=('valid_json', lambda x: (x==True).sum())
            ,invalid=('valid_json', lambda x: (x==False).sum())
            ,unique_reps=('repetition', 'nunique')
            )
            .reset_index()
        )
    

    grouped['complete_output'] = grouped['total_outputs'] == n_repetitions
    grouped['incomplete_output'] = grouped['total_outputs'] < n_repetitions

    summary = (
        grouped.groupby('model')
        .agg(
            claims_total=('id', 'count')
            ,complete_claims=('complete_output', 'sum')
            ,incomplete_claims=('incomplete_output', 'sum')
        ).reset_index()
    )

    print(f'\n {'-'*8} PER MODEL CLAIM COMPLETION SUMMARY: {'-'*8}')
    print(summary)
    print('-'*16)

    incomplete = grouped[grouped['incomplete_output'] == 1]
    if not incomplete.empty:
        print(f'\nINCOMPLETE (model, claim)')
        print(incomplete.groupby('model').agg(incomplete_counts = ('id', 'count')).reset_index())
    else:
        print(f'\nNO INCOMPLETE PAIRS FOR ALL MODELS')

    failed = grouped[grouped['invalid'] > 0]
    if not failed.empty:
        print(f'\nFAILED (model, claim)')
        print(failed.groupby('model').agg(failed_counts = ('invalid', 'sum')).reset_index())
        print(failed)
    else:
        print(f'\nNO FAILED (model, claim) PAIRS')


def load_results(model_names, dataset):
    dfs = [] 
    for model_n in model_names:
        for suffix in ("", "-failed"):
            path = Path(f'/home/rp-fril-mhpe/{model_n}-{args.dataset}{suffix}.csv')
            
            if not path.exists():
                print(f'File not found: {path}')
                continue

            
            df = pd.read_csv(path, low_memory=False)
            if suffix == '-failed':
                df['valid_json'] = False
                df['model'] = model_n
            
            df = df[['model', 'id', 'claim','repetition', 'valid_json']]
            dfs.append(df)
            
    combined = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    return combined



def main(args):
    profiles_root = yaml.safe_load(pathlib.Path('configs/models.yaml').read_text())
    profiles = profiles_root.get('profiles', {})
    model_names = list(profiles.keys())

    combined = load_results(model_names=model_names, dataset=args.dataset)
    
    check_results(combined, dataset_name=args.dataset, n_repetitions=10)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset',
                    help = 'Specify name of dataset',
                    default='sarcasm')
    
    args = ap.parse_args()
    main(args)
