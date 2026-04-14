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


def summarise_model_rates(grouped_df: pd.DataFrame) -> pd.DataFrame:
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
    positive = conf.positive_label
    negative = conf.negative_label

    # Group baseline outputs by model, (claim) id, and compute probability positive and negative label
    first_grouped = (
        first_df
        .groupby(['model', 'id'])
        .apply(lambda x: pd.Series({
            "p_pos": (x['label'] == positive).mean(),
            "p_neg": (x['label'] == negative).mean(),
        }))
        .reset_index()
    )

    # Create flag 'flip', to keep track whether the receiver changed its label to the label the sender proposed. 
    second_df['flip'] = (second_df['label_receiver_now'] == second_df['label_sender_before'])

    # Group by such that we get all cases and the p(label proposed by sender) over the 10 repetitions. 
    second_grouped = (
        second_df
        .groupby(['model_receiver', 'model_sender', 'label_receiver_before', 'label_sender_before', 'id', 'match_type'])['flip'].mean()
        .reset_index(name = 'p_round_2')
    )

    # Left join baseline probabilities. 
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


def get_delta_df_agree(first_df: pd.DataFrame, second_df: pd.DataFrame, conf: DatasetTaskSpec):
    pass


def main(args):
    base = Path(args.base_path)
    profiles_root = yaml.safe_load(Path("configs/models.yaml").read_text())
    profiles = profiles_root.get("profiles", {})
    model_names = list(profiles.keys())

    ds_config = DATASETS[args.dataset]

    # first_d = load_first_round_results(base, model_names, ds_config.dataset, failed=False)
    # first = load_all_as_dataframe(first_d)
    # discarded_claims = get_discarded_claims(ds_config.dataset, base)

    # discarded_pairs = discarded_claims[['model', 'id']].drop_duplicates()
    # discarded_pairs['_discard'] = True

    # # Removing discarded claims for given model
    # first = first.merge(discarded_pairs, on=['model', 'id'], how='left')
    # first = first[first['_discard'].isna()].drop(columns='_discard')

    # first_grouped = first.groupby(['model', 'id']).size().reset_index(name='count')

    # invalid = first_grouped[first_grouped['count'] != 10]
    # if not invalid.empty:
    #     print(f"WARNING: {len(invalid)} IDs with unexpected counts:\n{invalid}")

    # grouped_first = get_grouped_df(first, ds_config)
    # print(summarise_model_rates(grouped_df=grouped_first))

    # Delete later, this is just a check: 
    # for model in model_names: 
    #     df = pd.read_csv(f'/home/rp-fril-mhpe/input_round2/{ds_config.dataset}/{model}-self-interaction.csv')
    #     print(f'MODEL: {model}')
    #     print(df['match_type'].value_counts())

    # print(f'Macro-average positive rate: {grouped_first.groupby('model')['positive_rate'].mean()}')
    # print('Overall positive rates:')
    # prs = compute_overall_positive_rate(first, conf=ds_config)
    # print(prs)


    
    # plot_label_claim_distribution(grouped_df=grouped_first)
    second = load_all_as_dataframe(load_second_round_results(base, model_names, ds_config.dataset))

    second_grouped = second.groupby(['model_receiver', 'model_sender', 'label_receiver_before', 'label_sender_before', 'match_type']).size().reset_index(name='count')
    
    print(second_grouped[second_grouped['model_receiver'] == 'gemma-3-4b'])
    # print(second_grouped[second_grouped['count'] != 10])

    # print(first_grouped['count'].value_counts())
    # print(second_grouped['count'].value_counts())

    return 
    second_disagreeing = second[~second['match_type'].isin(['1:1', '0:0'])]
    
    # For disagreeing cases
    deltas_df = get_delta_df(first, second_disagreeing, ds_config)
    deltas_df.to_csv(f'deltas_{ds_config.dataset}.csv', index=False)


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
