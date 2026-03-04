import os
import random
import pandas as pd
from collections import defaultdict
from pathlib import Path
import argparse
from itertools import combinations

N_REPETITIONS = 10 

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

    grouped = df.groupby(["model", "id"])

    for (model, claim_id), group in grouped:
        labels = group["label"].tolist()
        # print(labels)
    
        explanations_by_label = {
            0: group[group["label"] == 'literal']["explanation"].tolist(),
            1: group[group["label"] == 'sarcastic']["explanation"].tolist(),
        }

        # print(explanations_by_label)
        unique_labels = set(labels)
        is_consistent = len(unique_labels) == 1
        consistent_label = list(unique_labels)[0] if is_consistent else None

        model_claim_dict[model][claim_id] = {
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

    sender_expls = sender_data["explanations_by_label"][label]
    receiver_expls = receiver_data["explanations_by_label"][label]

    for i in range(N_REPETITIONS):
        rows.append({
            "id": claim_id,
            
            "model_sender": sender,
            "model_receiver": receiver,
            "label_sender": label,
            "label_receiver": label,
            "explanation_sender": sender_expls[i],
            "explanation_receiver": receiver_expls[i],
        })

    return rows


def generate_disagree_case1(sender, receiver, claim_id, sender_data, receiver_data):
    """
    Both inconsistent.
    Generate:
        10 rows sender=1, receiver=0
        10 rows sender=0, receiver=1
    """
    rows = []

    sender_1 = sender_data["explanations_by_label"][1]
    sender_0 = sender_data["explanations_by_label"][0]
    receiver_1 = receiver_data["explanations_by_label"][1]
    receiver_0 = receiver_data["explanations_by_label"][0]

    # sender=1, receiver=0
    if sender_1 and receiver_0:
        s_sample = sample_with_replacement(sender_1, N_REPETITIONS)
        r_sample = sample_with_replacement(receiver_0, N_REPETITIONS)

        for i in range(N_REPETITIONS):
            rows.append({
                "model_sender": sender,
                "model_receiver": receiver,
                "id": claim_id,
                "label_sender": 1,
                "label_receiver": 0,
                "explanation_sender": s_sample[i],
                "explanation_receiver": r_sample[i],
            })

    # sender=0, receiver=1
    if sender_0 and receiver_1:
        s_sample = sample_with_replacement(sender_0, N_REPETITIONS)
        r_sample = sample_with_replacement(receiver_1, N_REPETITIONS)

        for i in range(N_REPETITIONS):
            rows.append({
                "model_sender": sender,
                "model_receiver": receiver,
                "id": claim_id,
                "label_sender": 0,
                "label_receiver": 1,
                "explanation_sender": s_sample[i],
                "explanation_receiver": r_sample[i],
            })

    return rows

def generate_disagree_case2A(sender, receiver, claim_id, sender_data, receiver_data):
    """
    Both consistent but opposite labels.
    No upsampling.
    """
    rows = []

    sender_label = sender_data["consistent_label"]
    receiver_label = receiver_data["consistent_label"]

    sender_expls = sender_data["explanations_by_label"][sender_label]
    receiver_expls = receiver_data["explanations_by_label"][receiver_label]

    for i in range(N_REPETITIONS):
        rows.append({
            "model_sender": sender,
            "model_receiver": receiver,
            "id": claim_id,
            "label_sender": sender_label,
            "label_receiver": receiver_label,
            "explanation_sender": sender_expls[i],
            "explanation_receiver": receiver_expls[i],
        })

    return rows


def generate_disagree_case2B(sender, receiver, claim_id, sender_data, receiver_data):
    """
    One consistent, one inconsistent.
    Upsample only inconsistent model.
    """
    rows = []

    # identify consistent vs inconsistent
    if sender_data["is_consistent"] and not receiver_data["is_consistent"]:
        consistent = sender
        inconsistent = receiver
        consistent_data = sender_data
        inconsistent_data = receiver_data
        flip = True  # swap roles later
    else:
        consistent = receiver
        inconsistent = sender
        consistent_data = receiver_data
        inconsistent_data = sender_data
        flip = False

    consistent_label = consistent_data["consistent_label"]
    opposite_label = 1 - consistent_label

    inconsistent_pool = inconsistent_data["explanations_by_label"][opposite_label]
    consistent_pool = consistent_data["explanations_by_label"][consistent_label]

    if not inconsistent_pool:
        return rows

    inconsistent_sample = sample_with_replacement(inconsistent_pool, N_REPETITIONS)

    for i in range(N_REPETITIONS):
        if flip:
            rows.append({
                "model_sender": sender,
                "model_receiver": receiver,
                "id": claim_id,
                "label_sender": consistent_label,
                "label_receiver": opposite_label,
                "explanation_sender": consistent_pool[i],
                "explanation_receiver": inconsistent_sample[i],
            })
        else:
            rows.append({
                "model_sender": sender,
                "model_receiver": receiver,
                "id": claim_id,
                "label_sender": opposite_label,
                "label_receiver": consistent_label,
                "explanation_sender": inconsistent_sample[i],
                "explanation_receiver": consistent_pool[i],
            })

    return rows



def process_all_pairs(model_claim_dict: dict):
    '''
    Writes all cases to file. 
    '''
    models = list(model_claim_dict.keys())

    for m1, m2 in combinations(models, 2):
        print(m1, m2)

        agree_rows = []
        disagree_rows = []

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
                    agree_rows.extend(
                        generate_agree_rows(
                            m1, m2, claim_id, m1_data, m2_data
                        )
                    )
                    continue
            # Disagree: Both models are consistent, but they disagree. 
            if (
                    m1_data["is_consistent"] and m2_data["is_consistent"] 
                    and m1_data["consistent_label"] != m1_data["consistent_label"]
                ):
                    disagree_rows.extend(
                        generate_disagree_case2A(
                            m1, m2, claim_id, m1_data, m2_data
                        )
                    )
            


            


    for receiver in models:
        for sender in models:

            sender_ids = set(model_claim_dict[sender].keys())
            receiver_ids = set(model_claim_dict[receiver].keys())
            shared_ids = sender_ids.intersection(receiver_ids)

            for claim_id in shared_ids:

                sender_data = model_claim_dict[sender][claim_id]
                receiver_data = model_claim_dict[receiver][claim_id]

                # -------- AGREE --------
                if (
                    sender_data["is_consistent"]
                    and receiver_data["is_consistent"]
                    and sender_data["consistent_label"]
                    == receiver_data["consistent_label"]
                ):
                    agree_rows.extend(
                        generate_agree_rows(
                            sender, receiver, claim_id, sender_data, receiver_data
                        )
                    )
                    continue

                # -------- DISAGREE --------
                if (
                    sender_data["is_consistent"]
                    and receiver_data["is_consistent"]
                    and sender_data["consistent_label"]
                    != receiver_data["consistent_label"]
                ):
                    disagree_rows.extend(
                        generate_disagree_case2A(
                            sender, receiver, claim_id, sender_data, receiver_data
                        )
                    )

                elif (
                    not sender_data["is_consistent"]
                    and not receiver_data["is_consistent"]
                ):
                    disagree_rows.extend(
                        generate_disagree_case1(
                            sender, receiver, claim_id, sender_data, receiver_data
                        )
                    )

                else:
                    disagree_rows.extend(
                        generate_disagree_case2B(
                            sender, receiver, claim_id, sender_data, receiver_data
                        )
                    )

            # -------- WRITE OUTPUT --------
            write_output(receiver, sender, "agree", agree_rows)
            write_output(receiver, sender, "disagree", disagree_rows)


# ============================================================
# STEP 4 — WRITE FILES
# ============================================================

def write_output(receiver, sender, category, rows):
    if not rows:
        return

    dir_path = os.path.join(OUTPUT_ROOT, receiver, category)
    os.makedirs(dir_path, exist_ok=True)

    file_path = os.path.join(dir_path, f"{sender}.csv")

    df = pd.DataFrame(rows)
    df.to_csv(file_path, index=False)


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





