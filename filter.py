import pandas as pd
import yaml 
from pathlib import Path

base = Path('/home/rp-fril-mhpe/second')

profiles_root = yaml.safe_load(Path('configs/models.yaml').read_text())
profiles = profiles_root.get('profiles', {})
models = list(profiles.keys())

for model in models:
    path = base / f'{model}-sarcasm.csv'
    if path.exists():
        df = pd.read_csv(path)
    else:
        continue


    has_self = (df['model_sender'] == model) & (df['model_receiver'] == model)
    if has_self.sum() > 0:
        print(f'MODEL: {model} has {has_self.sum()} self-interaction rows')

    else:
        print(f'No self-interaction for model: {model}')
