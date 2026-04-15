import argparse
import math
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import yaml
from utils.prompt_registry import DATASETS, DatasetTaskSpec

#TODO: Create overview over dropped claims.
#TODO: Input for round 2 (before and after )

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

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

def load_second_round_results(base: Path, models: list, dataset: str) -> dict[str, pd.DataFrame]:
    '''
    Load all model results for a dataset for second round.
    '''

    results = {}
    for model in models:
        dfs = []

        # main results 
        path = base / 'second' / f'{model}-{dataset}.csv'
        if path.exists():
            dfs.append(pd.read_csv(path))

        # self-interaction 
        if dataset == 'sarcasm': #TODO: change this when get all results for self-interaction 
            self_path = base / 'self' / 'second' / f'{model}-{dataset}.csv'
            if self_path.exists():
                dfs.append(pd.read_csv(self_path))

        if dfs:
            results[model] = pd.concat(dfs, ignore_index=True)
    if not results:
        raise ValueError(f'No second round data loaded for {dataset}')
    return results


def load_all_as_dataframe(df_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Load all first round results and tag with model column."""
    dfs = []
    for model, df in df_dict.items():
        print(f'looking at model {model}')
        dfs.append(df)
    
    #DEBUG: 
    print(f'number of dataframes: {len(dfs)}')
    return pd.concat(dfs, ignore_index=True)

def compute_overall_positive_rate(df: pd.DataFrame, conf: DatasetTaskSpec) -> pd.DataFrame:
    """
    Function to compute the overall prediction distribution.
    Takes the first round combined df. 
    I.e. with this we mean over all rows, NOT on a claim level.
    """
    df = df.copy()
    df['is_positive'] = df['label'] == conf.positive_label
    pr = df.groupby('model')['is_positive'].agg('mean').reset_index()
    
    return pr  

def get_discarded_claims(dataset: str, base: Path) -> pd.DataFrame:
    """
    Function to get df over all discarded claims, for all models on a given dataset. 
    """

    discarded_path = base / 'input_round2' / dataset / 'discarded.csv'

    return pd.read_csv(discarded_path)


def discarded_claims_to_latex(df: pd.DataFrame) -> str:
    grouped = (
        df.groupby('model')['id']
        .apply(lambda x: ', '.join(map(str, sorted(x))))
        .reset_index()
    )
    grouped['count'] = df.groupby('model')['id'].count().values

    rows = []
    for _, row in grouped.iterrows():
        rows.append(f"    {row['model']} & {row['id']} & {row['count']} \\\\")

    body = "\n\\midrule\n".join(rows)

    latex = f"""\\begin{{table}}[h]
                \\centering
                \\caption{{Claims discarded due to insufficient valid outputs ($<$10) per model.}}
                \\label{{tab:discarded_claims}}
                \\begin{{tabular}}{{lll}}
                \\toprule
                \\textbf{{Model}} & \\textbf{{Discarded Claim IDs}} & \\textbf{{Count}} \\\\
                \\midrule
                {body}
                \\bottomrule
                \\end{{tabular}}
                \\end{{table}}"""

    return latex

def validate_repetitions(df: pd.DataFrame, group_cols: list, expected: int = 10):
    counts = df.groupby(group_cols).size().reset_index(name='count')
    invalid = counts[counts['count'] != expected]

    if not invalid.empty:
        print(f"{len(invalid)} groups with unexpected counts:\n{invalid}")

def plot_label_claim_distribution(grouped_df: pd.DataFrame, dataset: str):
    """
    Plotting the positive rate distribution of results in round 1.
    """
    models = grouped_df['model'].unique()
    n_models = len(models)
    ncols = 2
    nrows = math.ceil(n_models / ncols)
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(16, 4 * nrows))
    axs_flat = axs.ravel()

    x_ticks = [round(x * 0.1, 1) for x in range(11)]

    for i, (model_name, ax) in enumerate(zip(models, axs_flat)):
        model_res = grouped_df[grouped_df['model'] == model_name]

        counts_perc = model_res['positive_rate'].value_counts(normalize=True).reindex(x_ticks, fill_value=0)*100

        # print(f'Model: {model_name}')
        # print(counts_perc)

        sns.barplot(x = counts_perc.index,
                    y = counts_perc.values,
                    ax=ax)

        ax.set_xticks(range(11))
        ax.set_xticklabels(x_ticks)
        
        # Letters from a-g 
        ax.text(-0.1, 1.05, f"{chr(97 + i)}", transform=ax.transAxes, fontsize=14, fontweight='bold', va='top', ha='right')

        ax.set_title(model_name, fontsize = 16)


        if i % ncols == 0:
            ax.set_ylabel('Percent', fontsize = 14)
        else:
            ax.set_ylabel("")

        if i // ncols == nrows - 1:
            ax.set_xlabel('Positive Rate', fontsize = 14)
        else:
            ax.set_xlabel("")
        
        ax.tick_params(labelsize=11)

        # This is putting values over the bars 
        # for p in ax.patches:
        #     height = p.get_height()
        #     if height > 0:
        #         ax.annotate(f'{height:.1f}%', (p.get_x() + p.get_width() / 2, height),
        #             ha='center', va='bottom', fontsize=9)
                
    for j in range(len(models), len(axs_flat)):
        axs_flat[j].set_visible(False)

    plt.tight_layout()
    sns.despine()
    save_path = f'plots/positive-rate-distr-{dataset}.png'
    print(f'Saving plots over distribution to file: {save_path}')
    plt.savefig(save_path, dpi=300, bbox_inches="tight")


def get_grouped_df(df: pd.DataFrame, conf: DatasetTaskSpec):
    '''
    Computes the positive rate on a claim-level.
    df
    '''

    df = df.copy()
    df['is_positive'] = df['label'] == conf.positive_label

    grouped = df.groupby(['model', 'id']).agg(positive_rate = ('is_positive', 'mean')).reset_index()
    return grouped


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
        )
    ).reset_index()


    # Create flag 'flip', to keep track whether the receiver changed its label to the label the sender proposed. 
    second_df['flip'] = (second_df['label_receiver_now'] == second_df['label_sender_before'])

    # Group by such that we get all cases and the p(label proposed by sender) over the 10 repetitions. 
    second_grouped = (
        second_df
        .groupby(['model_receiver', 'model_sender', 'label_receiver_before', 'label_sender_before', 'id', 'match_type'])['flip'].mean()
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
    
    per_match_type = delta_df.groupby(['model_receiver', 'model_sender', 'match_type']).agg(
        total_positive_delta = ('delta_positive_only', 'sum'),
        total_positive_budget = ('max_delta', 'sum'),
        total_negative_delta = ('delta_negative_only', 'sum'),
        total_negative_budget = ('max_delta_neg', 'sum'),
        possible_positive_count = ('possible_pos', 'sum'), 
        possible_negative_count = ('possible_neg', 'sum'),
        count = ('delta', 'size')
    ).reset_index()

    per_match_type['positive_delta_realisation'] = (
        per_match_type['total_positive_delta'] / 
        per_match_type['total_positive_budget'].replace(0, pd.NA)
    )
    per_match_type['negative_delta_realisation'] = (
        per_match_type['total_negative_delta'] / 
        per_match_type['total_negative_budget'].replace(0, pd.NA)
    )

    
    # Computing macro-averages 
    per_model_pair = per_match_type.groupby(['model_receiver', 'model_sender']).agg(
        macro_pos_delta_realisation = ('positive_delta_realisation', 'mean'),
        macro_neg_delta_realisation = ('negative_delta_realisation', 'mean'),
        count = ('count', 'size') 
    ).reset_index()

    return per_match_type, per_model_pair


def main(args):
    base = Path(args.base_path)
    profiles_root = yaml.safe_load(Path("configs/models.yaml").read_text())
    profiles = profiles_root.get("profiles", {})
    model_names = list(profiles.keys())

    ds_config = DATASETS[args.dataset]
    print(f'[DATASET] : {args.dataset}')

    print('Computing results from the first round....')
    first_d = load_first_round_results(base, model_names, ds_config.dataset, failed=False)
    first = load_all_as_dataframe(first_d)
    discarded_claims = get_discarded_claims(ds_config.dataset, base)

    discarded_pairs = discarded_claims[['model', 'id']].drop_duplicates()
    discarded_pairs['_discard'] = True

    # # Removing discarded claims for given model
    first = first.merge(discarded_pairs, on=['model', 'id'], how='left')
    first = first[first['_discard'].isna()].drop(columns='_discard')

    validate_repetitions(first, group_cols=['model', 'id'], expected=10)

    grouped_first = get_grouped_df(first, ds_config)

    print('Consistent / inconsistent labelling distribution')
    print(summarise_model_rates(grouped_df=grouped_first))
    
    print('Overall positive rate')
    print(compute_overall_positive_rate(first, conf=ds_config))

    print('"Model-bias", i.e. prediction distribution based on majority label.')
    majority_label_prop = grouped_first.groupby('model')['positive_rate'].apply(lambda x: (x >= 0.5).mean()).reset_index(name='proportion_positive')
    print(majority_label_prop)
    
    plot_label_claim_distribution(grouped_df=grouped_first, dataset=ds_config.dataset)
    

    print('Computing results from the second round...')
    second = load_all_as_dataframe(load_second_round_results(base, model_names, ds_config.dataset))


    # We group such that we know how many are in each sender, receiver, match_type 
    second_grouped = second.groupby(['model_receiver', 'model_sender', 'label_receiver_before', 'label_sender_before', 'match_type']).size().reset_index(name='count').sort_values(by='model_receiver')
    print('Counts of rows in cases')
    print(second_grouped)

    validate_repetitions(second, group_cols=['model_receiver', 'model_sender', 'label_receiver_before', 'label_sender_before', 'match_type', 'id'], expected=10)

    agreeing_input = second['match_type'].isin(['1:1', '0:0'])
    second_disagreeing = second[~agreeing_input]
    
    output_dir = Path('evaluation') / ds_config.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # For disagreeing cases
    deltas_df = get_delta_df(first, second_disagreeing, ds_config)
    deltas_df.to_csv(f'{output_dir}/deltas_disagreeing.csv', index=False)

    per_match_type, per_model_pair = summarise_deltas(deltas_df)
    per_match_type.to_csv(f'{output_dir}/deltas_match_type_disagreeing.csv', index=False)
    per_model_pair.to_csv(f'{output_dir}/deltas_model_disagreeing.csv', index=False)
    print('Summary of deltas for disagreeing cases:')
    print(per_model_pair)

    # For agreeing cases 
    second_agreeing = second[agreeing_input]
    deltas_df_agree = get_delta_df(first, second_agreeing, ds_config)
    deltas_df_agree.to_csv(f'{output_dir}/deltas_agreeing.csv', index=False)

    per_match_type, per_model_pair = summarise_deltas(deltas_df_agree)
    per_match_type.to_csv(f'{output_dir}/deltas_match_type_agreeing.csv', index=False)
    per_model_pair.to_csv(f'{output_dir}/deltas_model_agreeing.csv', index=False)
    print('Summary of deltas for agreeing cases:')
    print(per_model_pair)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_path',
                default='/home/rp-fril-mhpe',
                help='base path for results.')
    ap.add_argument("--dataset", 
                    help="Specify name of dataset",
                    default="sarcasm")
    args = ap.parse_args()
    main(args)
