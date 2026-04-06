import argparse
import math
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import yaml
from utils.prompt_registry import DATASETS, DatasetTaskSpec

#TODO: Micro-average label distribution per model.
#TODO: Positive-rate distribution, both metric and also plot.
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
        path = base / 'second' / f'{model}-{dataset}.csv'
        if path.exists():
            results[model] = pd.read_csv(path)
    
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

def compute_overall_positive_rate(df: pd.DataFrame, conf: DatasetTaskSpec) -> dict:
    """
    Function to compute the overall prediction distribution.
    Takes a single df for a single model. 
    I.e. with this we mean over all rows, NOT on a claim level.
    """

    return df.groupby('model')['label'].apply(lambda x: (x == conf.positive_label).sum() / len(x)).to_dict()

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

def plot_label_claim_distribution(grouped_df: pd.DataFrame):
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

        print(f'Model: {model_name}')
        print(counts_perc)

        sns.barplot(x = counts_perc.index,
                    y = counts_perc.values,
                    ax=ax)

        ax.set_xticks(range(11))
        ax.set_xticklabels(x_ticks)
        
        # Letters from a-g 
        ax.text(-0.1, 1.05, f"{chr(97 + i)}", transform=ax.transAxes, fontsize=14, fontweight='bold', va='top', ha='right')

        ax.set_title(model_name, fontsize = 13)


        if i % ncols == 0:
            ax.set_ylabel('Percent', fontsize = 12)
        else:
            ax.set_ylabel("")

        if i // ncols == nrows - 1:
            ax.set_xlabel('Positive Rate', fontsize = 12)
        else:
            ax.set_xlabel("")
        
        ax.tick_params(labelsize=11)

    for j in range(len(models), len(axs_flat)):
        axs_flat[j].set_visible(False)


    plt.tight_layout()
    sns.despine()
    print('Saving plots over distribution to file: "plots/label-dist-all.png')
    plt.savefig("plots/label-dist-all.png", dpi=300, bbox_inches="tight")


def get_grouped_df(df: pd.DataFrame, conf: DatasetTaskSpec):
    '''
    Computes the positive rate on a claim-level.
    df
    '''

    grouped = (df.groupby(['model', 'id'])['label']
               .apply(lambda x: (x == conf.positive_label).mean())
               .reset_index()
               .rename(columns={'label': 'positive_rate'}))
    
    return grouped


def get_delta_df(first_df: pd.DataFrame, second_df: pd.DataFrame, conf: DatasetTaskSpec) -> pd.DataFrame:
    '''
    Computes the delta dataframe, based on first and second round results. 
    '''
    positive = conf.positive_label
    negative = conf.negative_label


    first_grouped = (
        first_df
        .groupby(['model', 'id'])
        .apply(lambda x: pd.Series({
            "p_pos": (x['label'] == positive).mean(),
            "p_neg": (x['label'] == negative).mean(),
        }))
        .reset_index()
    )
    second_df['flip'] = (second_df['label_receiver_now'] == second_df['label_sender_before'])

    second_grouped = (
        second_df
        .groupby(['model_receiver', 'model_sender', 'label_receiver_before', 'label_sender_before', 'id', 'match_type'])['flip'].mean()
        .reset_index(name = 'p_round_2')
    )

    combined = second_grouped.merge(
        first_grouped,
        left_on=['model_receiver', 'id'],
        right_on=['model', 'id'],
        how='left'
    )


    influenced_towards_pos = combined['label_sender_before'] == positive

    combined['delta'] = (
        combined['p_round_2'] - combined['p_pos']
    ).where(influenced_towards_pos, combined['p_round_2'] - combined['p_neg'])


    combined['max_delta'] = (
        1 - combined['p_pos']
    ).where(influenced_towards_pos, 1-combined['p_neg'])

    print('True')
    print(combined[(abs(combined['delta']) > abs(combined['max_delta']))])
    # combined['delta'] = combined.apply(
    #     lambda r: r['p_round_2'] - r['p_pos']
    #     if r['label_sender_before'] == positive
    #     else r['p_round_2'] - r['p_neg'],
    #     axis=1
    # )    
    # combined['max_delta'] = combined.apply(
    #     lambda r: 1 - r['p_pos']
    #     if r['label_sender_before'] == positive
    #     else 1 - r['p_neg'],
    #     axis=1
    # )
    # Grouping all the B:B cases that have more than one row for receiver, sender, id
    # result = (
    #     combined
    #     .groupby(['model_receiver', 'model_sender', 'id'])
    #     .agg(
    #         delta_total = ('delta', 'sum'),
    #         max_delta_total = ('max_delta', 'sum')
    #     )
    #     .reset_index()
    # )

    return combined


def compute_delta_overall(delta_df: pd.DataFrame):

    # First we group by model_receiver, model_sender, match type, sum(delta)/sum(max_delta)
    agg = (
        delta_df.groupby(['model_receiver', 'model_sender', 'match_type'], as_index=False)
        .agg(
            sum_delta=('delta', 'sum'),
            sum_max_delta=('max_delta', 'sum'),
            count=('delta', 'count')
        )
    )

    agg['influence'] = agg['sum_delta'] / agg['sum_max_delta'].replace(0, pd.NA)

    macro = (
        agg.groupby(['model_receiver', 'model_sender'], as_index=False)
        .agg(influence = ('influence', 'mean'),
             count=('count', 'sum'))
    )
    macro['match_type'] = 'all'

    columns = ['model_receiver', 'model_sender', 'match_type', 'influence']

    result = pd.concat([
        agg[columns],
        macro[columns],
        ],
        ignore_index=True
        )
    

    ## Add count to rows. 
    ## Consider using 1-max_delta for the negative influences. 
    return result 


def main(args):
    base = Path(args.base_path)
    profiles_root = yaml.safe_load(Path("configs/models.yaml").read_text())
    profiles = profiles_root.get("profiles", {})
    model_names = list(profiles.keys())

    ds_config = DATASETS[args.dataset]

    first_d = load_first_round_results(base, model_names, ds_config.dataset, failed=False)
    first = load_all_as_dataframe(first_d)
    discarded_claims = get_discarded_claims(ds_config.dataset, base)

    discarded_pairs = discarded_claims[['model', 'id']].drop_duplicates()
    discarded_pairs['_discard'] = True

    # Removing discarded claims for given model
    first = first.merge(discarded_pairs, on=['model', 'id'], how='left')
    first = first[first['_discard'].isna()].drop(columns='_discard')

    # prs = compute_overall_positive_rate(first, conf=ds_config)
    # print(prs)

    # grouped_first = get_grouped_df(first, ds_config)
    # print(f'Macro-average positive rate: {grouped_first.groupby('model')['positive_rate'].mean()}')
    
    # plot_label_claim_distribution(grouped_df=grouped_first)

    second = load_all_as_dataframe(load_second_round_results(base, model_names, ds_config.dataset))

    second_disagreeing = second[~second['match_type'].isin(['1:1', '0:0'])]

    # first = pd.read_csv('src/test-first-gemma.csv')
    # second_disagreeing = pd.read_csv('src/test-second-gemma.csv')

    deltas_df = get_delta_df(first, second_disagreeing, ds_config)


    # Overall average of delta (denominator is the sum of max_delta.)
    delta_overall = compute_delta_overall(delta_df=deltas_df)
    print('Delta overall, all deltas, no filter')
    print(delta_overall)

    # Overall excluding any negative cases

    delta_pos = deltas_df[deltas_df['delta'] >= 0]
    delta_overall_pos = compute_delta_overall(delta_df=delta_pos)
    print('Delta overall, only positive deltas')
    print(delta_overall_pos)

    # Average of the negative cases
    delta_neg = deltas_df[deltas_df['delta'] < 0]
    delta_overall_neg = compute_delta_overall(delta_df=delta_neg)
    print('Delta overall, only negative deltas')
    print(delta_overall_neg)

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
