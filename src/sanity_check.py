import pandas as pd 
import argparse
import pathlib, yaml
from pathlib import Path 

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


###########
def load_input(model_names):
    dfs = [] 
    columns = ['id', 'claim', 'model_receiver', 'model_sender', 'label_receiver', 'label_sender', 'explanation_receiver', 'explanation_sender', 'match_type'] 



    for model_n in model_names:
        path = Path(f'/home/rp-fril-mhpe/input_round2/{model_n}_disagree.csv')
        if not path.exists():
            print(f'File not found: {path}')
            continue

        
        df = pd.read_csv(path, low_memory=False)
        print(f"Loaded file: {path}")
        df = df[columns]
        dfs.append(df)
            
    combined = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    return combined
#########

# def analyse_inputs(inputs, model_names, model_pairs):
#     for model in model_names:
#         print(f'*********************************************************'*2)
#         print('RECEIVER:', model)
        
#         slice = inputs[inputs['model_receiver'] == model]
#         for sender in model_pairs.get(model, []):
#             print(f'CONSIDERING SENDER: {sender}')
#             sender_slice = slice[slice['model_sender'] == sender]
#             match_counts = sender_slice['match_type'].value_counts()
#             print("_______________________________________")
#             print(f"Match type distribution for {model}:")
#             print(match_counts, "\n")

def analyse_inputs(inputs, model_names, model_pairs):
    labels = ['0', '1', 'B']

    for receiver in model_names:
        print("\n" + "="*80)
        print(f"% RECEIVER: {receiver}")
        print("="*80)

        slice_df = inputs[inputs['model_receiver'] == receiver]

        for sender in model_pairs.get(receiver, []):
            sender_slice = slice_df[slice_df['model_sender'] == sender]

            # Count match_types
            counts = sender_slice['match_type'].value_counts().to_dict()

            # Initialize 3x3 table
            table = {s: {r: 0 for r in labels} for s in labels}

            # Fill table
            for match, count in counts.items():
                try:
                    r_label, s_label = match.split(":")
                    if r_label in labels and s_label in labels:
                        table[s_label][r_label] = count
                except:
                    continue

            # ---- Generate LaTeX ----
            print(f"\n% Sender: {sender}")
            print("\\begin{table}[h!]")
            print("\\centering")
            print("\\begin{tabular}{c|ccc}")
            print("\\toprule")
            print(f"Sender $\\backslash$ Receiver & 0 & 1 & B \\\\")
            print("\\midrule")

            for s in labels:
                row = " & ".join(str(table[s][r]) for r in labels)
                print(f"{s} & {row} \\\\")

            print("\\bottomrule")
            print("\\end{tabular}")
            print(f"\\caption{{Match distribution: sender={sender}, receiver={receiver}}}")
            print("\\end{table}")

def main(args):
    profiles_root = yaml.safe_load(pathlib.Path('configs/models.yaml').read_text())
    profiles = profiles_root.get('profiles', {})
    model_names = list(profiles.keys())

    model_pairs = {"llama-3.3-70b": ["llama-3.1-8b", "qwen-2.5-72b", "gemma-3-27b", "gpt-oss-20b"], # matching with big models and family
                   "llama-3.1-8b": ["llama-3.3-70b"],
                   "qwen-2.5-72b": ["qwen-2.5-7b", "llama-3.3-70b", "gemma-3-27b", "gpt-oss-20b"], # only matching with same family
                   "qwen-2.5-7b": ["qwen-2.5-72b"],
                   "gemma-3-27b": ["gemma-3-4b", "llama-3.3-70b", "qwen-2.5-72b", "gpt-oss-20b"],
                   "gemma-3-4b": ["gemma-3-27b"],
                   "gpt-oss-20b": ["llama-3.3-70b", "qwen-2.5-72b", "gemma-3-27b"]}

    inputs = load_input(model_names=model_names)
    analyse_inputs(inputs, model_names=model_names, model_pairs=model_pairs)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset',
                    help = 'Specify name of dataset',
                    default='sarcasm')
    
    args = ap.parse_args()
    main(args)
