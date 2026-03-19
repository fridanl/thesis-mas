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
    columns = ['model', 'id', 'claim','repetition', 'valid_json', 'label'] 

    suffixes = ['-disagree', 'agree']

    for model_n in model_names:
        for suffix in suffixes:
            path = Path(f'/home/rp-fril-mhpe/{model_n}-{suffix}.csv')
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

def analyse_inputs(inputs, model_names):
    for model in model_names:
        slice = inputs[inputs['receiver'] == model]
        print(f'*********************************************************')
        print(f'Currently looking at receier: {model}')
        match_counts = slice['match_type'].value_counts()
        print("_______________________________________")
        print(f"Match type distribution for {model}:")
        print(match_counts, "\n")



def main(args):
    profiles_root = yaml.safe_load(pathlib.Path('configs/models.yaml').read_text())
    profiles = profiles_root.get('profiles', {})
    model_names = list(profiles.keys())

    inputs = load_input(model_names=model_names)
    analyse_inputs(inputs, model_names=model_names)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset',
                    help = 'Specify name of dataset',
                    default='sarcasm')
    
    args = ap.parse_args()
    main(args)
