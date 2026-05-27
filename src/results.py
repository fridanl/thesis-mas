import argparse
from pathlib import Path
from dataclasses import dataclass, field
import pandas as pd
import yaml
from utils.prompt_registry import DATASETS, DatasetTaskSpec

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)


@dataclass
class ExperimentSpec:
    '''Where to find data for a given experiment.'''
    second_dir: str                      # Subdir for 2nd round results 
    first_dir: str | None = None          # subdir for 1st round results, if applicable 
    model_suffixes: list[str] = field(default_factory=lambda: [''])
    models: list[str] = field(default_factory=list)  # override model list (empty = use models.yaml)
    include_self: bool = False             # whether to also load self/ second/ files

EXPERIMENTS: dict[str, ExperimentSpec] = {
    'main': ExperimentSpec(
        second_dir='second',
        first_dir='first',
        include_self=True,
    ),
    'swap': ExperimentSpec(
        second_dir='swap/second',
        first_dir=None,
        models=['gpt-oss-20b', 'llama-3.3-70b'],
    ),
    'temperature': ExperimentSpec(
        second_dir='temperature/second',
        first_dir='temperature/first',
        model_suffixes=['-high-temp', '-low-temp', ''],
        models=['llama-3.3-70b', 'qwen-2.5-72b'],
    ),
    'no-explanation': ExperimentSpec(
        second_dir='no_explanation/second',
        first_dir=None,
        models=['llama-3.3-70b', 'qwen-2.5-72b'],
    ),
    'no-history': ExperimentSpec(
        second_dir='no_history/second',
        first_dir=None,
        models=['llama-3.3-70b', 'qwen-2.5-72b'],
    ),
}



def load_first_round_results(base: Path, models: list, dataset: str, spec: ExperimentSpec, failed: bool = False) -> dict[str, pd.DataFrame]:
    '''
    Load all model results for a dataset from first round. 
    failed: indicates whether to load all failed examples instead. 
    '''   
    if spec.first_dir is None:
        return {}
    
    suffix = '-failed' if failed else ''
    results = {}
    for model in models:
        for model_suffix in spec.model_suffixes:
            path = base / spec.first_dir / f'{model}{model_suffix}-{dataset}{suffix}.csv'
            if path.exists():
                results[f'{model}{model_suffix}'] = pd.read_csv(path)
            else:
                print(f'[WARN] First-round file not found: {path}')
    return results


def load_second_round_results(base: Path, models: list, dataset: str, spec: ExperimentSpec) -> dict[str, pd.DataFrame]:
    '''
    Load all model results for a dataset for second round.
    '''

    if not spec.second_dir:
        return {}
    
    results = {}
    for model in models:
        dfs = []
        for model_suffix in spec.model_suffixes:
            path = base / spec.second_dir / f'{model}{model_suffix}-{dataset}.csv'
            if path.exists():
                dfs.append(pd.read_csv(path))
                print(f'Successfully loaded: {path}')
            else:
                print(f'[WARN] Not found: {path}')
        
        if spec.include_self:
            self_path = base / 'self' / 'second' / f'{model}-{dataset}.csv'
            if self_path.exists():
                dfs.append(pd.read_csv(self_path))
    
        if dfs:
            results[model] = pd.concat(dfs, ignore_index=True)

    if not results:
        raise ValueError(f'No second-round data loaded for dataset={dataset}, experiment second_dir={spec.second_dir}')
    return results


def load_all_as_dataframe(df_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Load all first round results and tag with model column."""
    dfs = list(df_dict.values())
    print(f'Number of dataframes: {len(dfs)}')
    return pd.concat(dfs, ignore_index=True)

def compute_overall_positive_rate(df: pd.DataFrame, conf: DatasetTaskSpec) -> pd.DataFrame:
    """
    Function to compute the overall prediction distribution.
    Takes the first round combined df. 
    I.e. with this we mean over all rows, NOT on a claim level.
    """
    df = df.copy()
    df['is_positive'] = df['label'] == conf.positive_label
    return df.groupby('model')['is_positive'].agg('mean').reset_index()
    

def get_discarded_claims(dataset: str, base: Path) -> pd.DataFrame:
    '''
    Function to get df over all discarded claims from first round, for all models on a given dataset. 
    '''
    discarded_path = base / 'input_round2' / dataset / 'discarded.csv'
    if discarded_path.exists():
        return pd.read_csv(discarded_path)
    print(f'[WARN] No discarded claims file found at {discarded_path}, skipping.')
    return pd.DataFrame()

def validate_repetitions(df: pd.DataFrame, group_cols: list, expected: int = 10):
    counts = df.groupby(group_cols).size().reset_index(name='count')
    invalid = counts[counts['count'] != expected]

    if not invalid.empty:
        print(f"{len(invalid)} groups with unexpected counts:\n{invalid}")

def get_grouped_df(df: pd.DataFrame, conf: DatasetTaskSpec) -> pd.DataFrame:
    '''
    Computes the positive rate on a claim-level.
    '''

    df = df.copy()
    df['is_positive'] = df['label'] == conf.positive_label

    return df.groupby(['model', 'id']).agg(positive_rate = ('is_positive', 'mean')).reset_index()


def summarise_model_rates(grouped_df: pd.DataFrame) -> pd.DataFrame:
    '''
    Input: Combined first-round df, that has been grouped on model, id and has positive_rate column. 
    Returns:
        Grouped df on model.
        Counts distribution consistent / inconsistent labelling for given id (claim).
    '''
    return (grouped_df.groupby('model')
            .apply(lambda g: pd.Series({
                'all_negative': (g['positive_rate'] == 0).sum(),
                'all_positive': (g['positive_rate'] == 1).sum(),
                'mixed':        ((g['positive_rate'] > 0) & (g['positive_rate'] < 1)).sum(),
            }))
            .reset_index())

def get_delta_df(first_df: pd.DataFrame, second_df: pd.DataFrame, conf: DatasetTaskSpec) -> pd.DataFrame:
    '''
    Computes the delta dataframe, based on first and second round results. 
    '''

    first_df = first_df.copy()
    second_df = second_df.copy()

    positive = conf.positive_label
    negative = conf.negative_label

    first_df['is_positive'] = first_df['label'] == positive
    first_df['is_negative'] = first_df['label'] == negative

    first_grouped = (
        first_df
        .groupby(['model', 'id'])
        .agg(
            p_pos = ('is_positive', 'mean'),
            p_neg = ('is_negative', 'mean')
        ).reset_index()
    )

    # Create flag 'flip', to keep track whether the receiver changed its label to the label the sender proposed. 
    second_df['flip'] = (second_df['label_receiver_now'] == second_df['label_sender_before'])

    # Group by such that we get all cases and the p(label proposed by sender) over the 10 repetitions. 
    second_grouped = (
        second_df
        .groupby(['model_receiver', 'model_sender', 'label_receiver_before', 'label_sender_before', 'id', 'match_type'])['flip']
        .mean()
        .reset_index(name = 'p_round_2')
    )

    # Left join baseline probabilities (to take into account, that we have subsampled for r2, and only some cases on first df is in second df)
    combined = second_grouped.merge(
        first_grouped,
        left_on=['model_receiver', 'id'],
        right_on=['model', 'id'],
        how='left'
    )

    # Flag direction of influence
    influenced_towards_pos = combined['label_sender_before'] == positive

    # Delta = interaction(label_sender) -  baseline p(label_sender)
    combined['delta'] = (
        combined['p_round_2'] - combined['p_pos']
    ).where(influenced_towards_pos, combined['p_round_2'] - combined['p_neg'])

    # max delta: The maximum a receiver can be influenced towards the proposed label, relative to baseline, i.e. reinforcing stance on proposed label. 
    combined['max_delta'] = (
        1 - combined['p_pos']
    ).where(influenced_towards_pos, 1-combined['p_neg'])

    # max_delta_neg = The maximum a receiver can be influenced/move away from the proposed label, relative to baseline, i.e. reinforcing stance on starting label. 
    combined['max_delta_neg'] = 1 - combined['max_delta']

    # Delta_clipped_neg: We clip any negative delta values to be 0.
    # We do this such, that we can keep the potential influence (i.e. max_delta) but count the influence as 'not achieved'.
    combined['delta_positive_only'] = combined['delta'].clip(lower = 0)

    # Delta_clipped_positive: We clip any positive delta values to be 0. 
    # Same reasoning as negative. 
    combined['delta_negative_only'] = combined['delta'].clip(upper = 0)

    return combined

def summarise_deltas(delta_df):
    '''
    Computes the macro average deltas per model pair and match type. 
    '''
    delta_df = delta_df.copy()
    delta_df['possible_neg'] = delta_df['max_delta_neg'] > 0
    delta_df['possible_pos'] = delta_df['max_delta'] > 0
    
    per_match_type = (
        delta_df
        .groupby(['model_receiver', 'model_sender', 'match_type'])
        .agg(
            total_positive_delta = ('delta_positive_only', 'sum'),
            total_positive_budget = ('max_delta', 'sum'),
            total_negative_delta = ('delta_negative_only', 'sum'),
            total_negative_budget = ('max_delta_neg', 'sum'),
            possible_positive_count = ('possible_pos', 'sum'), 
            possible_negative_count = ('possible_neg', 'sum'),
            count = ('delta', 'size')
        )
        .reset_index()
    )

    per_match_type['positive_delta_realisation'] = (
        per_match_type['total_positive_delta'] / 
        per_match_type['total_positive_budget'].replace(0, pd.NA)
    )
    per_match_type['negative_delta_realisation'] = (
        per_match_type['total_negative_delta'] / 
        per_match_type['total_negative_budget'].replace(0, pd.NA)
    )

    
    # Computing macro-averages 
    per_model_pair = (
        per_match_type
        .groupby(['model_receiver', 'model_sender'])
        .agg(
            macro_pos_delta_realisation = ('positive_delta_realisation', 'mean'),
            macro_neg_delta_realisation = ('negative_delta_realisation', 'mean'),
            count = ('count', 'sum') 
        )
        .reset_index()
    )

    return per_match_type, per_model_pair

def load_and_clean_first_round(base: Path, model_names: list, ds_config: DatasetTaskSpec, spec: ExperimentSpec) -> pd.DataFrame:
    '''
    Load first round results and remove discarded claims. 
    '''
    first_d = load_first_round_results(base, model_names, ds_config.dataset, spec, failed=False)
    first = load_all_as_dataframe(first_d)

    discarded_claims = get_discarded_claims(ds_config.dataset, base)
    if not discarded_claims.empty:
        discarded_pairs = discarded_claims[['model', 'id']].drop_duplicates()
        print('DISCARDED PAIRS')
        print(discarded_pairs)

        discarded_pairs = discarded_pairs.copy()
        discarded_pairs['_discard'] = True
        first = first.merge(discarded_pairs, on=['model', 'id'], how='left')
        first = first[first['_discard'].isna()].drop(columns='_discard')

    validate_repetitions(first, group_cols=['model', 'id'], expected=10)
    return first

def print_first_round_summary(first: pd.DataFrame, ds_config: DatasetTaskSpec):
    '''
    Print summary statistics for first round.
    '''
    grouped_first = get_grouped_df(first, ds_config)
    
    print('Consistent / inconsistent labelling distribution')
    model_rates = summarise_model_rates(grouped_df=grouped_first)
    print(model_rates)
    model_rates.to_csv(Path('evaluation') / ds_config.dataset / args.experiment / 'claims_label_consistency.csv')
    
    print('Overall positive rate')
    print(compute_overall_positive_rate(first, conf=ds_config))

    print('"Model-bias", i.e. prediction distribution based on majority label.')
    majority_label_prop = (
        grouped_first
        .groupby('model')['positive_rate']
        .apply(lambda x: (x >= 0.5).mean())
        .reset_index(name='proportion_positive')
    )
    print(majority_label_prop)

def load_and_clean_second_round(base: Path, model_names: list, ds_config: DatasetTaskSpec, spec: ExperimentSpec) -> pd.DataFrame:
    '''
    Load second round results and drop groups without exactly 10 repetitions.
    '''
    second = load_all_as_dataframe(load_second_round_results(base, model_names, ds_config.dataset, spec))

    group_cols = ['model_receiver', 'model_sender', 'label_receiver_before', 'label_sender_before', 'match_type', 'id']
    counts = second.groupby(group_cols)['id'].transform('size')
    # Only include ids with 10 repetitions 
    second = second[counts == 10]
    validate_repetitions(second, group_cols=group_cols, expected=10)
    return second

def compute_and_save_deltas(first: pd.DataFrame, second: pd.DataFrame, df_config: DatasetTaskSpec, output_dir: Path):
    '''
    Compute and save delta results for agreeing and disagreeing cases.
    '''
    agreeing_mask = second['match_type'].isin(['1:1', '0:0'])
    for label, subset in [('disagreeing', second[~agreeing_mask]), ('agreeing', second[agreeing_mask])]:
        deltas_df = get_delta_df(first, subset, df_config)
        deltas_df.to_csv(output_dir / f'deltas_{label}.csv', index=False)
        
        per_match_type, per_model_pair = summarise_deltas(deltas_df)
        per_match_type.to_csv(output_dir / f'deltas_match_type_{label}.csv', index=False)
        per_model_pair.to_csv(output_dir / f'deltas_model_{label}.csv', index=False)

        print(f'Summary of deltas for {label} cases:')
        print(per_model_pair)

def reverse_match_type(df: pd.DataFrame):
    # rename column; 'match_type' to match_type_old
    # make new match_type from match_type_old by reversing the string e.g. B:1 -> 1:B
    df = df.rename(columns={'match_type': 'match_type_old'})
    df['match_type'] = df['match_type_old'].str.split(':').str[::-1].str.join(':')
    return df


def main(args):
    base = Path(args.base_path)
    spec = EXPERIMENTS[args.experiment]

    if spec.models:
        model_names = spec.models
    else:
        profiles_root = yaml.safe_load(Path("configs/models.yaml").read_text())
        profiles = profiles_root.get("profiles", {})
        model_names = list(profiles.keys())
    
    print(f'[MODELS] : {model_names}')

    ds_config = DATASETS[args.dataset]
    print(f'[DATASET] : {args.dataset}')
    print(f'[EXPERIMENT] : {args.experiment} -> second_dir={spec.second_dir}, first_dir={spec.first_dir if spec.first_dir is not None else 'no first_dir given'}')

    if spec.first_dir is not None:
        print('\nComputing results from the first round....')
        first = load_and_clean_first_round(base, model_names, ds_config, spec)
        print_first_round_summary(first, ds_config)

    else:
        print(f'\n[INFO] : No first-round directory defined for experiment: {args.experiment}, skipping first-round analysis.')
        first = None 

    print('\nComputing results from the second round...')
    second = load_and_clean_second_round(base, model_names, ds_config, spec)
    if args.experiment == 'swap':
        second = reverse_match_type(second)
        print('Reversed match type...')
        print(second.head(3))

    if first is None:
        print(f'[INFO] : Loading main-experiment first-round results as baseline for delta computation...')
        main_spec = EXPERIMENTS['main']
        profiles_root = yaml.safe_load(Path("configs/models.yaml").read_text())
        all_models = list(profiles_root.get("profiles", {}).keys())
        first = load_and_clean_first_round(base, all_models, ds_config, main_spec)
    
    output_dir = Path('evaluation') / ds_config.dataset / args.experiment
    output_dir.mkdir(parents=True, exist_ok=True)    
    compute_and_save_deltas(first, second, ds_config, output_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_path',
                default='/home/rp-fril-mhpe',
                help='base path for results.')
    ap.add_argument("--dataset", 
                    help="Specify name of dataset",
                    default="sarcasm")
    ap.add_argument('--experiment', 
                    default='main',
                    choices=list(EXPERIMENTS.keys()),
                    help=f'Which experiment to run. Options: {list(EXPERIMENTS.keys())}')
    args = ap.parse_args()
    main(args)
