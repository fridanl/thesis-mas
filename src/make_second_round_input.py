import random
import pandas as pd
from collections import defaultdict
from pathlib import Path
import yaml
import argparse
from dataclasses import dataclass
from utils.prompt_registry import DATASETS


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

@dataclass
class TaskConfig:
    positive_label: str
    negative_label: str
    n_repetitions: int = 10
    random_seed: int = 42

def load_and_preprocess(df: pd.DataFrame, config: TaskConfig):
    '''
    Returns: 
    model_claim_dict = {
            model_name: {
                id: {
                    is_consistent: bool,
                    consistent_label: negative or positive or None,
                    explanations_by_label: {
                        [negative]: [...],
                        [positive]: [...]
                    },
                    labels: [10 values]
                } } }
    '''

    positive = config.positive_label
    negative = config.negative_label

    model_claim_dict = defaultdict(dict)

    model_discard, claim_discard = [], [] 
    grouped = df.groupby(["model", "id"])

    for (model, claim_id), group in grouped:
        if len(group) < config.n_repetitions: # if there are less repetitions than expected. 
            # write to file describing discarded examples.
            model_discard.append(model)
            claim_discard.append(claim_id)
            continue
        
        labels = group["label"].tolist()
    
        explanations_by_label = {
            negative: group[group["label"] == negative]["explanation"].tolist(),
            positive: group[group["label"] == positive]["explanation"].tolist(),
        }

        unique_labels = set(labels)
        is_consistent = len(unique_labels) == 1
        consistent_label = list(unique_labels)[0] if is_consistent else None
        claim_text = group['claim'].iloc[0]

        model_claim_dict[model][claim_id] = {
            "claim": claim_text,
            "is_consistent": is_consistent,
            "consistent_label": consistent_label,
            "explanations_by_label": explanations_by_label,
            "labels": labels,
        }
    
    df_discarded = pd.DataFrame(data = {'model': model_discard,'id': claim_discard})
    
    return model_claim_dict, df_discarded


def sample_with_replacement(pool, n):
    return [random.choice(pool) for _ in range(n)]


def generate_agree_rows(sender: str, receiver: str, claim_id: int, sender_data: dict, receiver_data: dict, config: TaskConfig):
    rows = []

    label = sender_data["consistent_label"]
    claim = sender_data['claim']

    sender_expls = sender_data["explanations_by_label"][label].copy()
    receiver_expls = receiver_data["explanations_by_label"][label].copy()
    
    if args.self_interaction:
        random.shuffle(sender_expls)
        random.shuffle(receiver_expls)

    label_bool = 1 if label == config.positive_label else 0 

    for i in range(config.n_repetitions):
        rows.append({
            "id": claim_id,
            'claim': claim,
            "model_receiver": receiver,
            "model_sender": sender,
            "label_receiver": label,
            "label_sender": label,
            "explanation_receiver": receiver_expls[i],
            "explanation_sender": sender_expls[i],
            "match_type": f'{label_bool}:{label_bool}'
        })

    return rows


def generate_disagree_rows(sender: str, receiver: str, claim_id: int, sender_data: dict, receiver_data: dict, config: TaskConfig):
    '''
      Catches all three cases of:
        Sender and receiver both have inconsistent labels -> We can match them up 1->0 and 0->1. Match_type: B-B
        Sender is inconsistent and receiver consistent -> We can only match them up consistent label of receiver -> 0/1.
        Sender is consistent and receiver inconsistent -> We can only match them up 0/1 -> consistent label of sender.

    The notation of ->:
        label_x -> label_y, denotes the direction the receiver-agent is attempted influenced in. 
        So label_x will be the receiver agent's label and label_y the sender-agent's label. 
    Generate:
        10 or 20 rows of label, explanation pairs. 
    
    '''
    rows = []
    claim = sender_data['claim']

    positive = config.positive_label
    negative = config.negative_label

    sender_unique_labels = [
        label for label in [negative, positive]
        if len(sender_data["explanations_by_label"][label]) > 0
    ]
    receiver_unique_labels = [
        label for label in [negative, positive] 
        if len(receiver_data['explanations_by_label'][label]) > 0
    ]

    def encode_labels(labels_list):
        if set(labels_list) == {negative, positive}:
            return 'B'
        elif set(labels_list) == {negative}:
            return '0'
        elif set(labels_list) == {positive}:
            return '1'
        
    sender_code = encode_labels(sender_unique_labels)
    receiver_code = encode_labels(receiver_unique_labels)

    match_type_base = f'{receiver_code}:{sender_code}'
    # sender_label -> receiver_label
    directions = [
        (positive, negative),
        (negative, positive)
    ]

    for sender_label, receiver_label in directions:
        sender_pool = sender_data['explanations_by_label'][sender_label]
        receiver_pool = receiver_data['explanations_by_label'][receiver_label]

        if len(sender_pool) == 0 or len(receiver_pool) == 0: # if either has not predicted to the label 
            continue 
        
        s_sample = sample_with_replacement(sender_pool, n=config.n_repetitions)
        r_sample = sample_with_replacement(receiver_pool, n=config.n_repetitions)

        for i in range(config.n_repetitions):
            rows.append({
                "id": claim_id,
                "claim": claim,
                "model_receiver": receiver,
                "model_sender": sender,
                "label_receiver": receiver_label,
                "label_sender": sender_label,
                "explanation_receiver": r_sample[i],
                "explanation_sender": s_sample[i],
                'match_type': match_type_base
            })

    return rows



def process_all_pairs(model_claim_dict: dict, receiver: str, config: TaskConfig):
    '''
    Writes all matches of one receiver to all other models to file.  
    '''
    models = list(model_claim_dict.keys())

    if args.self_interaction: # matching the models up with themselves only, when self_interaction is set to True
        model_pairs = {"llama-3.3-70b": ["llama-3.3-70b"], # ORIGINAL
                    "llama-3.1-8b": ["llama-3.1-8b"],
                    "qwen-2.5-72b": ["qwen-2.5-72b"],
                    "qwen-2.5-7b": ["qwen-2.5-7b"],
                    "gemma-3-27b": ["gemma-3-27b"],
                    "gemma-3-4b": ["gemma-3-4b"],
                    "gpt-oss-20b": ["gpt-oss-20b"]}

    else:
        # a fixed set of model pairs that we chose to match up, so we don't get every possible pair
        # only within family and the large models across family
        model_pairs = {"llama-3.3-70b": ["llama-3.1-8b", "qwen-2.5-72b", "gemma-3-27b", "gpt-oss-20b"],
                   "llama-3.1-8b": ["llama-3.3-70b"],
                   "qwen-2.5-72b": ["qwen-2.5-7b", "llama-3.3-70b", "gemma-3-27b", "gpt-oss-20b"], 
                   "qwen-2.5-7b": ["qwen-2.5-72b"],
                   "gemma-3-27b": ["gemma-3-4b", "llama-3.3-70b", "qwen-2.5-72b", "gpt-oss-20b"],
                   "gemma-3-4b": ["gemma-3-27b"],
                   "gpt-oss-20b": ["llama-3.3-70b", "qwen-2.5-72b", "gemma-3-27b"]}

    # Dicts for agree and disagree rows sender model
    agree_rows_for_receiver: list[dict] = []
    disagree_rows_for_receiver: list[dict] = []

    # Ids of the rows in dataset, that has been processed by the receiver. 
    receiver_ids = set(model_claim_dict[receiver].keys())
    # Loop over all possible models to match up with. 
    for sender in models:    
        if sender == receiver and not args.self_interaction:
            continue # skip matching up with itself if self_interaction is false
        if sender not in model_pairs.get(receiver, []):  #making sure we only take the fixed pairs
            continue
        sender_ids = set(model_claim_dict[sender].keys())
        shared_ids = sender_ids & receiver_ids

        for i in shared_ids:
            receiver_data = model_claim_dict[receiver][i]
            sender_data = model_claim_dict[sender][i]

            if (
            receiver_data["is_consistent"] and sender_data["is_consistent"] # Both models have consistent labels
            and 
            receiver_data["consistent_label"] == sender_data["consistent_label"] # and equal labels 
            ):
                agree_rows_for_receiver.extend(generate_agree_rows(sender = sender, receiver= receiver, claim_id=i, sender_data=sender_data, receiver_data=receiver_data, config=config))

            else: # everything else is disagreement / mixed / inconsistent 
                disagree_rows_for_receiver.extend(generate_disagree_rows(sender=sender, receiver=receiver, claim_id=i, sender_data=sender_data, receiver_data=receiver_data, config=config))

    if disagree_rows_for_receiver:
        df_disagree = pd.DataFrame(disagree_rows_for_receiver)
        df_disagree = df_disagree.sort_values(["id", "model_receiver", "model_sender"]).reset_index(drop=True)
    else: 
        df_disagree = pd.DataFrame()

    if agree_rows_for_receiver:
        df_agree = pd.DataFrame(agree_rows_for_receiver)
        df_agree = df_agree.sort_values(["id", "model_receiver", "model_sender"]).reset_index(drop=True)
    else:
        df_agree = pd.DataFrame()

    return df_agree, df_disagree
    
def main(args):
    dfs = [] 
    path = Path(args.input_dir)
    files = [f for f in path.glob("*.csv") if not f.name.endswith("-failed.csv")] # loading all the files in the given input folder, ignoring failed

    for i, file in enumerate(files):
        if not file.exists():
            print(f'File not found: {file}')
            continue
        print(f'Loading file: {file}')
        df = pd.read_csv(file, low_memory=False)
        dfs.append(df)


    combined = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    dataset_spec = DATASETS[args.dataset]
    t_config = TaskConfig(
        positive_label=dataset_spec.positive_label,
        negative_label=dataset_spec.negative_label
    )
    random.seed(t_config.random_seed)

    outdir = args.output_root 
    model_claim_dict, discard = load_and_preprocess(combined, t_config)
    if not args.self_interaction:
        discard.to_csv(f'{outdir}/{args.dataset}/discarded.csv', index=False)                                  
        print(f'Saving the discarded claims to {outdir}/{args.dataset}/discarded.csv')

    model_names = list(combined['model'].unique())
    for receiver in model_names:
        if receiver not in model_claim_dict.keys():
            continue
        agree, disagree = process_all_pairs(model_claim_dict=model_claim_dict, receiver=receiver, config=t_config)
        
        if not args.self_interaction:
            if not agree.empty:
                print(f'Saving {len(agree)} lines to {f'{outdir}/{args.dataset}/{receiver}_agree.csv'}')
                agree.to_csv(f'{outdir}/{args.dataset}/{receiver}_agree.csv', index=False) 
            if not disagree.empty:                         
                print(f'Saving {len(disagree)} lines to {f'{outdir}/{args.dataset}/{receiver}_disagree.csv'}')
                disagree.to_csv(f'{outdir}/{args.dataset}/{receiver}_disagree.csv', index=False)                    
        else:
            if not agree.empty:
                agree.to_csv(f'{outdir}/{args.dataset}/{receiver}_self_interaction_agree.csv', index=False)

            if not disagree.empty:
                disagree.to_csv(f'{outdir}/{args.dataset}/{receiver}_self_interaction_disagree.csv', index=False)     

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset',
                    help = 'Specify name of dataset',
                    default='sarcasm')
    ap.add_argument('--output_root',
                    help='Path of root to save files',
                    default="/home/rp-fril-mhpe/input_round2")
    ap.add_argument('--input_dir',
                    help='Path of root to load the files',
                    default="/home/rp-fril-mhpe/first")
    ap.add_argument('--self_interaction',
                    help='If set to true, a dataset for self interaction is created from B:B instances',
                    action='store_true')
    args = ap.parse_args()
    main(args)
