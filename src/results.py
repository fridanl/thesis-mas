import argparse
import math
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
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

def compute_overall_positive_rate(df: pd.DataFrame, conf: DatasetTaskSpec) -> float:
    """
    Function to compute the overall prediction distribution.
    Takes a single df for a single model. 
    I.e. with this we mean over all rows, NOT on a claim level.
    """

    positive_count = (df['label'] == conf.positive_label).sum()

    return positive_count / df.shape[0]

def get_discarded_claims(dataset: str, base: Path) -> pd.DataFrame:
    """
    Function to get df over all discarded claims, for all models on a given dataset. 
    """

    discarded_path = base / 'input_round2' / dataset / 'discarded.csv'

    return pd.read_csv(discarded_path)



# def check_results(combined, *, dataset_name, n_repetitions):
#     """Function to check if the output in files (valid and failed) correspond to expected number."""
#     datasets = {"sarcasm": "data/sarc/sarcasm.csv"}

#     if dataset_name not in datasets:
#         raise ValueError(f"Unknown dataset: {dataset_name}")

#     path_data = Path(datasets[dataset_name])
#     data = pd.read_csv(path_data, low_memory=False)

#     n_claims = data.shape[0]
#     expected_output_size = n_claims * n_repetitions

#     print(f"[DATASET]: {dataset_name}")
#     print(f"[CLAIMS IN DATASET]: {n_claims}")
#     print(f"[REPETITIONS PER CLAIM]: {n_repetitions}")
#     print(f"[EXPECTED ROWS PER MODEL]: {expected_output_size}")

#     print(f"\n {'-' * 8} PER MODEL CHECK {'-' * 8}")
#     print(f"[SIZE OF OUTPUT PER MODEL, GROUPED BY VALID, FAILED]:")
#     output_sizes = (
#         combined.groupby(["model", "valid_json"])
#         .agg(output_size=("id", "size"))
#         .reset_index()
#     )
#     print(output_sizes)

#     grouped = (
#         combined.groupby(["model", "id"])
#         .agg(
#             total_outputs=("id", "size"),
#             valid_outputs=("valid_json", lambda x: (x == True).sum()),
#             invalid=("valid_json", lambda x: (x == False).sum()),
#             unique_reps=("repetition", "nunique"),
#         )
#         .reset_index()
#     )

#     # Complete and incomplete outputs in terms of number of valid + number of invalid
#     grouped["complete_output"] = grouped["total_outputs"] == n_repetitions
#     grouped["incomplete_output"] = grouped["total_outputs"] < n_repetitions

#     summary = (
#         grouped.groupby("model")
#         .agg(
#             claims_total=("id", "count"),
#             complete_claims=("complete_output", "sum"),
#             incomplete_claims=("incomplete_output", "sum"),
#         )
#         .reset_index()
#     )

#     print(f"\n {'-' * 8} PER MODEL CLAIM COMPLETION SUMMARY: {'-' * 8}")
#     print(summary)
#     print("-" * 16)

#     incomplete = grouped[grouped["incomplete_output"] == 1]
#     if not incomplete.empty:
#         print(f"\nINCOMPLETE (model, claim)")
#         print(
#             incomplete.groupby("model")
#             .agg(incomplete_counts=("id", "count"))
#             .reset_index()
#         )
#     else:
#         print(f"\nNO INCOMPLETE PAIRS FOR ALL MODELS")

#     failed = grouped[grouped["invalid"] > 0]
#     if not failed.empty:
#         print(f"\nFAILED (model, claim)")
#         print(
#             failed.groupby("model").agg(failed_counts=("invalid", "sum")).reset_index()
#         )
#         # print(failed)
#     else:
#         print(f"\nNO FAILED (model, claim) PAIRS")


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

def plot_label_claim_distribution(grouped_dfs: dict[str, pd.DataFrame]):
    """
    Plotting the positive rate distribution of results in round 1.
    """
    models = list(grouped_dfs.keys())
    n_models = len(models)
    ncols = 2
    nrows = math.ceil(n_models / ncols)
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(16, 4 * nrows), sharey=True)
    axs_flat = axs.ravel()

    x_ticks = [round(x * 0.1, 1) for x in range(11)]

    for i, (model_name, ax) in enumerate(zip(models, axs_flat)):
        model_res = grouped_dfs[model_name]


        sns.histplot(data=model_res, 
                     ax=ax, 
                     x="positive_rate", 
                     stat='percent',
                     discrete=True)

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks, rotation=45)
        ax.set_xlim(-0.15, 1.15)
        
        # Letters from a-g 
        ax.text(-0.05, 1.05, f"{chr(97 + i)}", transform=ax.transAxes, fontsize=14, fontweight='bold', va='top', ha='right')

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
    plt.savefig("plots/label-dist-all.png", dpi=300, bbox_inches="tight")


def get_grouped_df(df: pd.DataFrame, conf: DatasetTaskSpec):
    '''
    Computes the positive rate on a claim-level.
    df:
        df: For specific model.
    '''

    grouped = (df.groupby(['model', 'id'])['label']
               .apply(lambda x: (x == conf.positive_label).mean())
               .reset_index()
               .rename(columns={'label': 'positive_rate'}))
    
    return grouped


def main(args):
    base = Path(args.base_path)
    profiles_root = yaml.safe_load(Path("configs/models.yaml").read_text())
    profiles = profiles_root.get("profiles", {})
    model_names = list(profiles.keys())

    ds_config = DATASETS[args.dataset]

    first_d = load_first_round_results(base, model_names, ds_config.dataset, failed=False)
    discarded_claims = get_discarded_claims(ds_config.dataset, base)

    grouped_d = {}

    for model in model_names:
        print(f'LOOKING AT MODEL: {model}')
        first_raw = first_d[model]
        discarded = discarded_claims[discarded_claims['model'] == model]['id'].to_list()
        first = first_raw[~first_raw['id'].isin(discarded)]
        print(f'Dropped {first_raw.shape[0]-first.shape[0]} rows, due to at least one failed attempt for claim.')
        pr = compute_overall_positive_rate(first, conf=ds_config)
        print(f'Overall positive rate: {pr}')
        grouped = get_grouped_df(first, ds_config)
        print(f'Macro-average positive rate: {grouped['positive_rate'].mean()}')

        grouped_d[model] = grouped

    # print(discarded_claims_to_latex(discarded_claims))    
    plot_label_claim_distribution(grouped_dfs=grouped_d)


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
