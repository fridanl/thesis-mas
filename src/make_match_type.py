import pandas as pd
from pathlib import Path

def add_match_type_to_gpt(
    base: Path,
    dataset: str = 'sarcasm',
    model: str = 'gpt-oss-20b',
    self_interaction: bool = False
):
    # Load second round results
    if self_interaction:
        second_path = base / 'self' / 'second' / f'{model}-{dataset}.csv'
    else:
        second_path = base / 'second' / f'{model}-{dataset}.csv'

    # second_path = base / 'second' / f'{model}-{dataset}_no_match.csv'
    second = pd.read_csv(second_path)
    print(f'Loaded {len(second)} rows from {second_path}')

    if self_interaction:
        # Load and concatenate agree and disagree subsampled input files
        agree_path = base / 'subsampled_input_round2' / dataset / 'gpt' / dataset / 'self' /f'{model}_self_interaction_agree_subsampled.csv'
        disagree_path = base / 'subsampled_input_round2' / dataset / 'gpt' / dataset / 'self' / f'{model}_self_interaction_disagree_subsampled.csv'
    else:
        # Load and concatenate agree and disagree subsampled input files
        agree_path = base / 'subsampled_input_round2' / dataset / 'gpt' / dataset / f'{model}_agree_subsampled.csv'
        disagree_path =base / 'subsampled_input_round2' / dataset / 'gpt' / dataset / f'{model}_disagree_subsampled.csv'

    if agree_path.exists():
        agree = pd.read_csv(agree_path)
    else:
        agree = pd.DataFrame()
    disagree = pd.read_csv(disagree_path)

    print(f'Agree rows: {len(agree)}')
    print(f'Disagree rows: {len(disagree)}')
    input_df = pd.concat([agree, disagree], ignore_index=True)
    print(f'Loaded {len(input_df)} rows from subsampled input files')

    # Keep only the columns needed for joining + match_type
    join_cols = ['id', 'model_sender', 'model_receiver', 'label_sender_before', 'label_receiver_before', 'match_type']

    # Some input files use label_sender / label_receiver instead of label_sender_before / label_receiver_before
    # Rename if necessary
    rename_map = {
        'label_sender': 'label_sender_before',
        'label_receiver': 'label_receiver_before',
    }
    input_df = input_df.rename(columns={k: v for k, v in rename_map.items() if k in input_df.columns})

    input_lookup = input_df[join_cols].drop_duplicates()
    print(f'Unique join key combinations in input: {len(input_lookup)}')

    # Left join match_type onto second round results
    merge_on = ['id', 'model_sender', 'model_receiver', 'label_sender_before', 'label_receiver_before']
    second = second.merge(input_lookup, on=merge_on, how='left')

    # CHECK 1: Rows missing match_type after join
    missing = second[second['match_type'].isna()]
    if missing.empty:
        print('\n[CHECK 1] All rows have a match_type after join.')
    else:
        print(f'\n[CHECK 1] {len(missing)} rows are missing match_type after join:')
        print(missing)

    # CHECK 2: Count occurrences grouped by the validation group cols
    group_cols = ['model_receiver', 'model_sender', 'label_receiver_before', 'label_sender_before', 'match_type', 'id']
    counts = (
        second.groupby(group_cols, dropna=False)
        .size()
        .reset_index(name='count')
    )
    print('\n[CHECK 2] Row counts per group:')
    print(counts)
    print(f'\nCount distribution:\n{counts["count"].value_counts().sort_index()}')
    
    agree_check = second[second['match_type'].isin(['1:1','0:0'])]
    if agree_check.empty:
        print('[WARN] There are no agree cases in this input.')
    else:
        print(f'[CHECK 3] There are: {len(agree_check)} agree cases.') 

    return second


if __name__ == '__main__':

    # self gpt sarcasm 
    base = Path('/home/rp-fril-mhpe/')
    second = add_match_type_to_gpt(base, dataset='sarcasm', self_interaction=True)
    second.to_csv(base / 'self' / 'second' / 'gpt-oss-20b-sarcasm-match.csv', index=False)
    print(f'Saving file gpt self sarcasm file to: {base}/self/second/gpt-oss-20b-sarcasm-match.csv')

    # self gpt commonsense
    base = Path('/home/rp-fril-mhpe/')
    second = add_match_type_to_gpt(base, dataset='commonsense', self_interaction=True)
    second.to_csv(base / 'self' / 'second' / 'gpt-oss-20b-commonsense-match.csv', index=False) 
    print(f'Saving file gpt self commonsense file to: {base}/second/gpt-oss-20b-commonsense-match.csv')

    base = Path('/home/rp-fril-mhpe/')
    second = add_match_type_to_gpt(base, dataset='commonsense')
    second.to_csv(base / 'second' / 'gpt-oss-20b-commonsense-match.csv', index=False)
    print(f'Saving file gpt commonsense file to: {base}/second/gpt-oss-20b-commonsense-match.csv')