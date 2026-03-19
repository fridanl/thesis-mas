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

    suffixes = ['disagree', 'agree']

    for model_n in model_names:
        for suffix in suffixes:
            path = Path(f'/home/rp-fril-mhpe/input_round2/{model_n}_{suffix}.csv')
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

def analyse_inputs(inputs, model_names, model_pairs):
    for model in model_names:
        print(f'*********************************************************'*2)
        print('RECEIVER:', model)
        
        slice = inputs[inputs['model_receiver'] == model]
        for sender in model_pairs.get(model, []):
            print(f'CONSIDERING SENDER: {sender}')
            sender_slice = slice[slice['model_sender'] == sender]
            match_counts = sender_slice['match_type'].value_counts()
            print("_______________________________________")
            print(f"Match type distribution for {model}:")
            print(match_counts, "\n")



def main(args):
    profiles_root = yaml.safe_load(pathlib.Path('configs/models.yaml').read_text())
    profiles = profiles_root.get('profiles', {})
    model_names = list(profiles.keys())

    model_pairs = {"llama-3.3-70b": ["llama-3.1-8b", "qwen-2.5-72b", "gemma-3-27b"], # matching with big models and family
                   "llama-3.1-8b": ["llama-3.3-70b"],
                   "qwen-2.5-72b": ["qwen-2.5-7b", "llama-3.3-70b", "gemma-3-27b"], # only matching with same family
                   "qwen-2.5-7b": ["qwen-2.5-72b"],
                   "gemma-3-27b": ["gemma-3-4b", "llama-3.3-70b", "qwen-2.5-72b"],
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
