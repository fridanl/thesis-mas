import pandas as pd
from pathlib import Path

def add_match_type_to_gpt(
    base: Path,
    dataset: str = 'sarcasm',
    model: str = 'gpt-oss-20b',
    self_interaction: bool = False):

    # Load second round results
    if self_interaction:
        second_path = base / 'self' / 'second' / f'{model}-{dataset}-self_interaction_agree_no_match.csv'
    else:
        second_path = base / 'second' / f'{model}-{dataset}-agree_no_match.csv'

    # second_path = base / 'second' / f'{model}-{dataset}_no_match.csv'
    second = pd.read_csv(second_path)
    print(f'Loaded {len(second)} rows from {second_path}')

    if self_interaction:
        if dataset == 'commonsense':
        # Load and concatenate agree and disagree subsampled input files
            agree_path = base / 'subsampled_input_round2' / dataset / 'gpt' / dataset / 'self' / dataset / f'{model}_self_interaction_agree_subsampled.csv'
            disagree_path = base / 'subsampled_input_round2' / dataset / 'gpt' / dataset / 'self' / dataset /f'{model}_self_interaction_disagree_subsampled.csv'
        else:
            agree_path = base / 'subsampled_input_round2' / dataset / f'{model}_self_interaction_agree_subsampled.csv'
            disagree_path = base / 'subsampled_input_round2' / dataset / f'{model}_self_interaction_disagree_subsampled.csv'

    else:
        if dataset == 'commonsense':
            # Load and concatenate agree and disagree subsampled input files
            agree_path = base / 'subsampled_input_round2' / dataset / 'gpt' / dataset / f'{model}_agree_subsampled.csv'
            disagree_path =base / 'subsampled_input_round2' / dataset / 'gpt' / dataset / f'{model}_disagree_subsampled.csv'

        else:
            agree_path = base / 'subsampled_input_round2' / dataset / f'{model}_agree_subsampled.csv'
            disagree_path =base / 'subsampled_input_round2' / dataset / f'{model}_disagree_subsampled.csv'

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


def concat(df, self_interaction):
    base = Path('/home/rp-fril-mhpe/')
    # Load existing file
    if not self_interaction:
        existing_path = base / 'second' / 'gpt-oss-20b-sarcasm.csv'
    else:
        existing_path = base / 'self' / 'second' / 'gpt-oss-20b-sarcasm.csv' 

    existing_df = pd.read_csv(existing_path)
    print(f'Rows in original df: {len(existing_df)}')

    # Match type df 
    print(f'Rows in match_type df: {len(df)}')

    # Concat and save
    combined_df = pd.concat([existing_df, df], ignore_index=True)
    print(f'Rows in combined df: {len(combined_df)}')

    if not self_interaction:
        output_path = base / 'second' / 'gpt-oss-20b-sarcasm_new.csv'
    else:
        output_path = base / 'self' / 'second' / 'gpt-oss-20b-sarcasm_new.csv' 
    
    combined_df.to_csv(output_path, index=False)
    print(f'Saving combined file to: {output_path}')

def make_qwen_llama_data():
    """
    for robustness experiments
    """
    print('Constructing dataset for llama as receiver...')
    df = pd.read_csv('/home/rp-fril-mhpe/subsampled_input_round2/sarcasm/llama-3.3-70b_disagree_subsampled.csv')
    subset = df[df['model_sender'] == 'qwen-2.5-72b']
    print(subset['match_type'].value_counts())
    subset.to_csv('/home/rp-fril-mhpe/subsampled_input_round2/no_history/sarcasm/llama-3.3-70b_disagree_subsampled.csv', index=False)
    print('Saved file to /home/rp-fril-mhpe/subsampled_input_round2/no_history/sarcasm/llama-3.3-70b_disagree_subsampled.csv')

    print('Constructing dataset for qwen as receiver...')
    df2 = pd.read_csv('/home/rp-fril-mhpe/subsampled_input_round2/sarcasm/qwen-2.5-72b_disagree_subsampled.csv')
    subset2 = df2[df2['model_sender'] == 'llama-3.3-70b']
    print(subset2['match_type'].value_counts())
    subset2.to_csv('/home/rp-fril-mhpe/subsampled_input_round2/no_history/sarcasm/qwen-2.5-72b_disagree_subsampled.csv', index=False)


def make_llama_data():

    out_dir = Path("/home/rp-fril-mhpe/tmp/llama")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = ["llama-3.3-70b-sentiment.csv"]

    input_dir = Path("/home/rp-fril-mhpe/second")

    for file in files:
        df = pd.read_csv(input_dir / file)

        before = len(df)
        remove_count = (df["model_sender"] == "llama-3.1-8b").sum()

        df = df[df["model_sender"] != "llama-3.1-8b"]

        after = len(df)

        print(f"{file}")
        print(f"  rows before:  {before}")
        print(f"  rows removed: {remove_count}")
        print(f"  rows after:   {after}")
        print()

        df.to_csv(out_dir / file, index=False)

    
if __name__ == '__main__':

    # self gpt sarcasm 
    # base = Path('/home/rp-fril-mhpe/')
    # second = add_match_type_to_gpt(base, dataset='sarcasm', self_interaction=True)
    # second.to_csv(base / 'self' / 'second' / 'gpt-oss-20b-sarcasm-match.csv', index=False)
    # print(f'Saving file gpt self sarcasm file to: {base}/self/second/gpt-oss-20b-sarcasm-match.csv')

    # # self gpt commonsense
    # base = Path('/home/rp-fril-mhpe/')
    # second = add_match_type_to_gpt(base, dataset='commonsense', self_interaction=True)
    # second.to_csv(base / 'self' / 'second' / 'gpt-oss-20b-commonsense-match.csv', index=False) 
    # print(f'Saving file gpt self commonsense file to: {base}/second/gpt-oss-20b-commonsense-match.csv')

    # gpt commonsense
    # base = Path('/home/rp-fril-mhpe/')
    # second = add_match_type_to_gpt(base, dataset='commonsense')
    # second.to_csv(base / 'second' / 'gpt-oss-20b-commonsense-match.csv', index=False)
    # print(f'Saving file gpt commonsense file to: {base}/second/gpt-oss-20b-commonsense-match.csv')

    # make_qwen_llama_data()


    # gpt sentiment self
    # base = Path('/home/rp-fril-mhpe/')
    # second_self = add_match_type_to_gpt(base, dataset='sentiment', self_interaction=True)
    # second_self.to_csv(base / 'self' / 'second' / 'gpt-oss-20b-sentiment.csv', index=False) 
    # print(f'Saving file gpt self sentiment file to: {base}/self/second/gpt-oss-20b-sentiment.csv')


    # # gpt sentiment
    # second = add_match_type_to_gpt(base, dataset='sentiment', self_interaction=False)
    # second.to_csv(base / 'second' / 'gpt-oss-20b-sentiment.csv', index=False) 
    # print(f'Saving file gpt sentiment file to: {base}/second/gpt-oss-20b-sentiment.csv')

    #gpt agreeing
    #base = Path('/home/rp-fril-mhpe/')
    #second = add_match_type_to_gpt(base, dataset='sarcasm', self_interaction=False)
    #second.to_csv(base / 'second' / 'gpt-oss-20b-sarcasm_agree.csv', index=False) 
    #print(f'Saving file gpt sentiment file to: {base}/second/gpt-oss-20b-sarcasm_agree.csv')
    #concat(second, self_interaction=False)

    # gpt agreeing, self-interaction 
    #second_self = add_match_type_to_gpt(base, dataset='sarcasm', self_interaction=True)
    #second_self.to_csv(base / 'self' / 'second' / 'gpt-oss-20b-sarcasm_agree.csv', index=False) 
    #print(f'Saving file gpt sentiment file to: {base}/self/second/gpt-oss-20b-sarcasm_agree.csv')
    #concat(second_self, self_interaction=True)
    make_llama_data()