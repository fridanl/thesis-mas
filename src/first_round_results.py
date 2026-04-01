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

def compute_prediction_distribution(df: pd.DataFrame, conf: DatasetTaskSpec) -> float:
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





def check_results(combined, *, dataset_name, n_repetitions):
    """Function to check if the output in files (valid and failed) correspond to expected number."""
    datasets = {"sarcasm": "data/sarc/sarcasm.csv"}

    if dataset_name not in datasets:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    path_data = Path(datasets[dataset_name])
    data = pd.read_csv(path_data, low_memory=False)

    n_claims = data.shape[0]
    expected_output_size = n_claims * n_repetitions

    print(f"[DATASET]: {dataset_name}")
    print(f"[CLAIMS IN DATASET]: {n_claims}")
    print(f"[REPETITIONS PER CLAIM]: {n_repetitions}")
    print(f"[EXPECTED ROWS PER MODEL]: {expected_output_size}")

    print(f"\n {'-' * 8} PER MODEL CHECK {'-' * 8}")
    print(f"[SIZE OF OUTPUT PER MODEL, GROUPED BY VALID, FAILED]:")
    output_sizes = (
        combined.groupby(["model", "valid_json"])
        .agg(output_size=("id", "size"))
        .reset_index()
    )
    print(output_sizes)

    grouped = (
        combined.groupby(["model", "id"])
        .agg(
            total_outputs=("id", "size"),
            valid_outputs=("valid_json", lambda x: (x == True).sum()),
            invalid=("valid_json", lambda x: (x == False).sum()),
            unique_reps=("repetition", "nunique"),
        )
        .reset_index()
    )

    # Complete and incomplete outputs in terms of number of valid + number of invalid
    grouped["complete_output"] = grouped["total_outputs"] == n_repetitions
    grouped["incomplete_output"] = grouped["total_outputs"] < n_repetitions

    summary = (
        grouped.groupby("model")
        .agg(
            claims_total=("id", "count"),
            complete_claims=("complete_output", "sum"),
            incomplete_claims=("incomplete_output", "sum"),
        )
        .reset_index()
    )

    print(f"\n {'-' * 8} PER MODEL CLAIM COMPLETION SUMMARY: {'-' * 8}")
    print(summary)
    print("-" * 16)

    incomplete = grouped[grouped["incomplete_output"] == 1]
    if not incomplete.empty:
        print(f"\nINCOMPLETE (model, claim)")
        print(
            incomplete.groupby("model")
            .agg(incomplete_counts=("id", "count"))
            .reset_index()
        )
    else:
        print(f"\nNO INCOMPLETE PAIRS FOR ALL MODELS")

    failed = grouped[grouped["invalid"] > 0]
    if not failed.empty:
        print(f"\nFAILED (model, claim)")
        print(
            failed.groupby("model").agg(failed_counts=("invalid", "sum")).reset_index()
        )
        # print(failed)
    else:
        print(f"\nNO FAILED (model, claim) PAIRS")


def plot_label_claim_distribution(df, kde=True):
    """
    Plotting the positive rate distribution of results in round 1.
    """
    models = df["model"].unique()

    n_models = len(models)
    ncols = 2
    nrows = math.ceil(n_models / ncols)
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(16, 4 * nrows))

    for model_name, ax in zip(models, axs.ravel()):
        model_res = df[df["model"] == model_name].copy()

        if kde:
            sns.kdeplot(
                data=model_res,
                ax=ax,
                x="positive_rate",
                fill=True,
            )
        else:
            sns.histplot(data=model_res, ax=ax, x="positive_rate")

        ax.set_title(f"{model_name}")
        ax.set_xlabel("Positive Rate")

    plt.tight_layout()
    sns.despine()
    plt.savefig("plots/label-dist-all.png", dpi=300, bbox_inches="tight")


def label_distribution(df):

    # Overall label distribution for models
    grouped_model = (
        df[["model", "label"]]
        .groupby("model")
        .agg(positive_count_overall=("label", lambda x: (x == "sarcastic").sum()))
    )
    grouped_model = grouped_model["positive_count_overall"] / 1374540

    print("Label distribution over models")
    print(grouped_model)

    # Per model, claim
    grouped = (
        df.groupby(["model", "id"])
        .agg(
            valid_outputs=("valid_json", lambda x: (x == True).sum()),
            positive_count=("label", lambda x: (x == "sarcastic").sum()),
        )
        .reset_index()
    )

    grouped["positive_rate"] = grouped["positive_count"] / grouped["valid_outputs"]

    grouped_pr = grouped.groupby(["model", "positive_rate"]).size().reset_index()
    print("Grouped per model, id, positive rate")
    print(grouped_pr)

    #  print('Positive rate per model/claim')
    #  print(grouped)
    return grouped


def load_results(
    model_names: list[str], dataset: str, with_failed: bool
) -> pd.DataFrame:
    """ """
    dfs = []
    columns = ["model", "id", "claim", "repetition", "valid_json", "label"]

    if with_failed:
        suffixes = ("", "-failed")
    else:
        suffixes = ("",)

    for model_n in model_names:
        for suffix in suffixes:
            path = Path(f"/home/rp-fril-mhpe/{model_n}-{dataset}{suffix}.csv")

            if not path.exists():
                print(f"File not found: {path}")
                continue

            df = pd.read_csv(path, low_memory=False)
            if suffix == "-failed":
                df["valid_json"] = False
                df["model"] = model_n
                df["label"] = None

            df = df[columns]
            dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    return combined


def main(args):
    base = Path(args.base_path)
    profiles_root = yaml.safe_load(Path("configs/models.yaml").read_text())
    profiles = profiles_root.get("profiles", {})
    model_names = list(profiles.keys())

    ds_config = DATASETS[args.dataset]

    first = load_first_round_results(base, model_names, ds_config.dataset, failed=False)
    discarded_claims = get_discarded_claims(ds_config.dataset, base)


    print(discarded_claims)


    for model in model_names:
        pass 

    # compute_prediction_distribution(combined_all)


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
