import os
import random
import pandas as pd
from collections import defaultdict
from pathlib import Path
import argparse
from itertools import combinations
import pprint
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

def load_and_preprocess(df: pd.DataFrame, config: TaskConfig) -> dict:
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

    grouped = df.groupby(["model", "id"])

    for (model, claim_id), group in grouped:
        labels = group["label"].tolist()
        # print(labels)
    
        explanations_by_label = {
            negative: group[group["label"] == negative]["explanation"].tolist(),
            positive: group[group["label"] == positive]["explanation"].tolist(),
        }

        # print(explanations_by_label)
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
    # print(model_claim_dict)
    return model_claim_dict


def sample_with_replacement(pool, n):
    return [random.choice(pool) for _ in range(n)]


def generate_agree_rows(sender: str, receiver: str, claim_id: int, sender_data: dict, receiver_data: dict, config: TaskConfig):
    rows = []

    label = sender_data["consistent_label"]
    claim = sender_data['claim']

    sender_expls = sender_data["explanations_by_label"][label]
    receiver_expls = receiver_data["explanations_by_label"][label]
    label_bool = 1 if label == config.positive_label else 0 


    assert len(sender_expls) == len(receiver_expls) == config.n_repetitions
    for i in range(config.n_repetitions):
        rows.append({
            "id": claim_id,
            'claim': claim,
            "model_sender": sender,
            "model_receiver": receiver,
            "label_sender": label,
            "label_receiver": label,
            "explanation_sender": sender_expls[i],
            "explanation_receiver": receiver_expls[i],
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

    match_type_base = f'{receiver_code}-{sender_code}'
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
                "model_sender": sender,
                "model_receiver": receiver,
                "label_sender": sender_label,
                "label_receiver": receiver_label,
                "explanation_sender": s_sample[i],
                "explanation_receiver": r_sample[i],
                'match_type': match_type_base 
            })

    return rows



def process_all_pairs(model_claim_dict: dict, receiver: str, config: TaskConfig):
    '''
    Writes all matches of one receiver to all other models to file.  
    '''
    models = list(model_claim_dict.keys())

    # Dicts for agree and disagree rows sender model
    agree_rows_for_receiver: list[dict] = []
    disagree_rows_for_receiver: list[dict] = []

    # Ids of the rows in dataset, that has been processed by the receiver. 
    receiver_ids = set(model_claim_dict[receiver].keys())
    # Loop over all possible models to match up with. 
    for sender in models:
        if sender == receiver:
            continue
        sender_ids = set(model_claim_dict[sender].keys())
        shared_ids = receiver_ids.intersection(sender_ids) # only the examples that they both have answered 

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



    df_disagree = pd.DataFrame(disagree_rows_for_receiver)
    df_agree = pd.DataFrame(agree_rows_for_receiver)
    # print(df_agree.columns)
    # print(df_disagree.columns)

    df_agree = df_agree.sort_values(["id", "model_sender", "model_receiver"]).reset_index(drop=True)
    df_disagree = df_disagree.sort_values(["id", "model_sender", "model_receiver"]).reset_index(drop=True)

    print('agree')
    pprint.pprint(df_agree)
    print('disagree')
    pprint.pprint(df_disagree)

    # For test-data-r1.csv 
    # ID 1: Both models agree on claim, so that should be in the agree data. 
    # ID 2: Both models are consistent but disagree, 10 cases here per model. 
    # ID 3: Gemma is inconsistent, Llama is consistent, also 10 cases here per model.
    # ID 4: Llama is inconsistent, gemma is consistent, also 10 cases here per model. 
    # ID 5: Both models are inconsistent, 20 cases here per model. 
    
def main(args):
    # profiles_root = yaml.safe_load(Path('configs/models.yaml').read_text())
    # profiles = profiles_root.get('profiles', {})
    # model_names = list(profiles.keys())

    # dfs = [] 
    # for model_n in model_names:
    #     path = Path(f'/home/rp-fril-mhpe/{model_n}-{args.dataset}.csv')
    #     if not path.exists():
    #         print(f'File not found: {path}')
    #         continue
    #     df = pd.read_csv(path, low_memory=False)
    #     dfs.append(df)
            
    # combined = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    
    combined = pd.read_csv('test-data-r1.csv')
    dataset_spec = DATASETS[args.dataset]
    t_config = TaskConfig(
        positive_label=dataset_spec.positive_label,
        negative_label=dataset_spec.negative_label
    )
    random.seed(t_config.random_seed)

    model_claim_dict = load_and_preprocess(combined, t_config)
    # print(model_claim_dict)
    
    process_all_pairs(model_claim_dict=model_claim_dict, receiver=args.receiver, config=t_config)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset',
                    help = 'Specify name of dataset',
                    default='sarcasm')
    ap.add_argument('--output_root',
                    help='Path of root to save files',
                    default="thesis-mas/input_round2")
    ap.add_argument('--receiver', 
                    help='Specify the name of the receiver model, for which you want to create input for second round.')
    
    args = ap.parse_args()
    main(args)





