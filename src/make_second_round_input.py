import os
import random
import pandas as pd
from collections import defaultdict
from pathlib import Path
import argparse
from itertools import combinations
import pprint

N_REPETITIONS = 10
POSITIVE_LABEL = 'sarcastic'
NEGATIVE_LABEL = 'literal'

def load_and_preprocess(df: pd.DataFrame) -> dict:
    '''
    Returns: 
    model_claim_dict = {
            model_name: {
                id: {
                    is_consistent: bool,
                    consistent_label: 0 or 1 or None,
                    explanations_by_label: {
                        0: [...],
                        1: [...]
                    },
                    labels: [10 values]
                } } }
    '''

    model_claim_dict = defaultdict(dict)

    # Import dataset object and map sarcastic : 1, and literal: 0, and so on for other datasets
    # 1 = sarcastic, positive 
    # 0 = literal, negative

    grouped = df.groupby(["model", "id"])

    for (model, claim_id), group in grouped:
        labels = group["label"].tolist()
        # print(labels)
    
        explanations_by_label = {
            NEGATIVE_LABEL: group[group["label"] == NEGATIVE_LABEL]["explanation"].tolist(),
            POSITIVE_LABEL: group[group["label"] == POSITIVE_LABEL]["explanation"].tolist(),
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


def generate_agree_rows(sender, receiver, claim_id, sender_data, receiver_data):
    rows = []

    label = sender_data["consistent_label"]
    claim = sender_data['claim']

    sender_expls = sender_data["explanations_by_label"][label]
    receiver_expls = receiver_data["explanations_by_label"][label]
    
    # TODO: sampling up to 10 explanations or 
    # TODO: Check if len(explanations == N_repetitions)
    for i in range(N_REPETITIONS):
        rows.append({
            "id": claim_id,
            'claim': claim,
            "model_sender": sender,
            "model_receiver": receiver,
            "label_sender": label,
            "label_receiver": label,
            "explanation_sender": sender_expls[i],
            "explanation_receiver": receiver_expls[i],
        })

    return rows


def generate_disagree_rows(sender, receiver, claim_id, sender_data, receiver_data):
    """
    Catches all three cases of:
        Sender and receiver both have inconsistent labels -> We can match them up 1->0 and 0->1.
        Sender is inconsistent and receiver consistent -> We can only match them up consistent label of receiver -> 0/1.
        Sender is consistent and receiver inconsistent -> We can only match them up 0/1 -> consistent label of sender.

    The notation of ->:
        label_x -> label_y, denotes the direction the receiver-agent is attempted influenced in. 
        So label_x will be the receiver agent's label and label_y the sender-agent's label. 
    Generate:
        10 or 20 rows of label, explanation pairs. 
    """
    rows = []
    claim = sender_data['claim']

    sender_1 = sender_data["explanations_by_label"][POSITIVE_LABEL]
    sender_0 = sender_data["explanations_by_label"][NEGATIVE_LABEL]
    receiver_1 = receiver_data["explanations_by_label"][POSITIVE_LABEL]
    receiver_0 = receiver_data["explanations_by_label"][NEGATIVE_LABEL]

    # sender=1, receiver=0
    if len(sender_1) > 0  and len(receiver_0) > 0:
        s_sample = sample_with_replacement(sender_1, N_REPETITIONS)
        r_sample = sample_with_replacement(receiver_0, N_REPETITIONS)

        for i in range(N_REPETITIONS):
            rows.append({
                "id": claim_id,
                'claim': claim,
                "model_sender": sender,
                "model_receiver": receiver,
                "label_sender": POSITIVE_LABEL,
                "label_receiver": NEGATIVE_LABEL,
                "explanation_sender": s_sample[i],
                "explanation_receiver": r_sample[i],
            })

    # sender=0, receiver=1
    if len(sender_0) > 0 and len(receiver_1) > 0:
        s_sample = sample_with_replacement(sender_0, N_REPETITIONS)
        r_sample = sample_with_replacement(receiver_1, N_REPETITIONS)

        for i in range(N_REPETITIONS):
            rows.append({
                "id": claim_id,
                'claim': claim,
                "model_sender": sender,
                "model_receiver": receiver,
                "label_sender": NEGATIVE_LABEL,
                "label_receiver": POSITIVE_LABEL,
                "explanation_sender": s_sample[i],
                "explanation_receiver": r_sample[i],
            })

    return rows

def process_all_pairs(model_claim_dict: dict):
    '''
    Writes all cases to file. 
    '''
    models = list(model_claim_dict.keys())

    # Dicts for agree and disagree rows sender model
    agree_rows_for_receiver = defaultdict(list)
    disagree_rows_for_receiver = defaultdict(list)

    for m1, m2 in combinations(models, 2):
        print(m1, m2)

        m1_ids = set(model_claim_dict[m1].keys())
        m2_ids = set(model_claim_dict[m2].keys())
        shared_ids = m1_ids.intersection(m2_ids)

        for claim_id in shared_ids:
            m1_data = model_claim_dict[m1][claim_id]
            m2_data = model_claim_dict[m2][claim_id]

            # The two models are consistent in their labelling and they agree on the label 
            if (
                m1_data["is_consistent"] and m2_data["is_consistent"] # Both models have consistent labels
                and m1_data["consistent_label"] == m2_data["consistent_label"] # and equal labels 
                ):
                    agree_rows_for_receiver[m1].extend(
                        generate_agree_rows(
                            sender = m2, receiver= m1, claim_id=claim_id, sender_data=m2_data, receiver_data=m1_data
                        )
                    )
                    agree_rows_for_receiver[m2].extend(
                        generate_agree_rows(
                            sender = m1, receiver= m2, claim_id=claim_id, sender_data=m1_data, receiver_data=m2_data
                        )
                    )
                    continue
                    # Disagree: Both models are consistent, but they disagree. 
            if (
                    m1_data["is_consistent"] and m2_data["is_consistent"] 
                    and m1_data["consistent_label"] != m2_data["consistent_label"]
                ):
                    disagree_rows_for_receiver[m1].extend(
                        generate_disagree_rows(
                            sender = m2, receiver = m1, claim_id=claim_id, sender_data=m2_data, receiver_data=m1_data
                        )
                    )
                    disagree_rows_for_receiver[m2].extend(
                        generate_disagree_rows(
                            sender = m1, receiver = m2, claim_id=claim_id, sender_data=m1_data, receiver_data=m2_data
                        )
                    )
            elif (     # One model is consistent and another is not -> there is disagreement, but we can only match them up one direction disagree wise. 
                        #! Maybe add agree cases here as well, i.e. add agreement into agree dict. 
                    (m1_data["is_consistent"] and not m2_data["is_consistent"]) 
                    or 
                    (m2_data["is_consistent"] and not m1_data["is_consistent"])
                ):
                    disagree_rows_for_receiver[m2].extend(
                            generate_disagree_rows(
                                sender=m1, receiver=m2, claim_id=claim_id, sender_data=m1_data, receiver_data=m2_data
                            )
                        )
                    

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
    combined = pd.read_csv('../test-data-r1.csv')

    RANDOM_SEED = 42
    N_REPETITIONS = 10
    random.seed(RANDOM_SEED)
    model_claim_dict = load_and_preprocess(combined)
    process_all_pairs(model_claim_dict)



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset',
                    help = 'Specify name of dataset',
                    default='sarcasm')
    ap.add_argument('--output_root',
                    help='Path of root to save files',
                    default="thesis-mas/input_round2")
    
    args = ap.parse_args()
    main(args)





