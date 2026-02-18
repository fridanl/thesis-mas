from typing import List, Dict, Optional, Iterator
from itertools import islice
import pandas as pd
import json, csv, pathlib
from .prompts import SYSTEM_JSON_GUIDED_R1, USER_R1, SYSTEM_JSON_GUIDED_R2, USER_R2

def load_claims_text(path: str) -> List[Dict[str, str]]:
    items = []
    with open(path, 'r', encoding='utf-8') as file: 
        for i, line in enumerate(file):
            claim = line.strip()
            if not claim:
                continue
            
            items.append({'id': i, 'text': claim.strip('"').strip("'")})
    return items 


def load_claims_batches(
                path: str, 
                start: int=0,
                batch_size: int=256,
                limit: Optional[int] = None ) -> Iterator[List[Dict[str, str]]]:
    
    data = pd.read_csv(path)

    if start >= len(data):
        raise ValueError('Start is larger than size of data.')
    
    end = len(data) if limit is None else min(len(data), start+limit)
    
    if end <= start:
        return #nothing to yield 

    window = data.iloc[start:end]

    for i in range(start, len(window), batch_size):
        chunk = window.iloc[i : i+batch_size]
        buf = chunk.to_dict()
        
        buf = [r.to_dict() for _, r in chunk.iterrows()]

        yield buf

def build_conversations(
    examples: List[Dict[str, str]],
    system_prompt: str = SYSTEM_JSON_GUIDED_R1,
    user_template: str = USER_R1) -> List[List[Dict[int, str]]]:

    '''
    Several conversations will be a list of lists containing a dict for each user. 
    '''
    convs: List[List[Dict[str, str]]] = []

    for ex in examples:
        
       convs.append([
          {'role': 'system', 'content': system_prompt},
          {'role': 'user', 'content': user_template.format(claim=ex['text'])},
       ])

    return convs 

def build_conversations_round2(
    examples: List[Dict[int, str]],
    system_prompt: str = SYSTEM_JSON_GUIDED_R2,             # this can be changed in run eval to R2
    user_template: str = USER_R2) -> List[List[Dict[str, str]]]:

    '''
    Several conversations will be a list of lists containing a dict for each user.
    This is for round 2, so we append label and explanation from the previous round. 
    '''
    convs: List[List[Dict[str, str]]] = []

    for ex in examples:
        
        convs.append([
          {'role': 'system', 'content': system_prompt},
          {'role': 'user', 'content': user_template.format(claim=ex['claim'], label_sender=ex['label_sender'], explanation_sender=ex['explanation_sender'])},
          ])

    return convs 

def write_jsonl(records: List[Dict[str, Any]], path: pathlib.Path):
    with path.open('a', encoding='utf-8') as file: 
        for r in records:
            file.write(json.dumps(r, ensure_ascii=False) + '\n')

def _ensure_oneline(s: str) -> str:
    if s is None:
        return ""

    return json.dumps(s, ensure_ascii=False)


def write_csv(records: List[Dict[str, Any]], path: pathlib.Path, fields):
    # checking if file needs header 
    needs_header = (not path.exists()) or (path.stat().st_size == 0)
    with path.open('a', newline='', encoding='utf-8', errors='replace') as file:
        w = csv.DictWriter(file, fieldnames=fields)
        if needs_header:
            w.writeheader()
        for r in records:
            r['raw_text'] = _ensure_oneline(r['raw_text'])
            w.writerow({k: r.get(k) for k in fields})
            