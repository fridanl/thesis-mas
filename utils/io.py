from typing import List, Dict, Optional, Iterator, Any, Hashable, TypedDict, Sequence, Literal
import pandas as pd
import json, csv, pathlib
from datetime import datetime
import logging 

class ChatCompletionMessageParam(TypedDict):
    role: str
    content: str


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
                batch_size: int = 10,
                start: int = 0,
                limit: Optional[int] = None ) -> Iterator[List[Dict[Hashable, Any]]]:
    
    remaining = limit 
    
    skip = range(1, start+1) if start > 0 else None
    reader = pd.read_csv(path,
                         usecols=['id', 'text'],
                         chunksize=batch_size,
                         skiprows=skip,
                         low_memory=False)

    for chunk in reader: 
        if remaining is not None:
            if remaining <= 0:
                return 
            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining]
            remaining -= len(chunk)

        yield chunk.to_dict(orient = 'records')


def build_conversations(
    examples: List[Dict[Hashable, Any]],
    *,
    system_prompt: str,
    user_template: str,
    history: bool,
    round: Literal[1, 2]
    ) -> Sequence[List[ChatCompletionMessageParam]]:

    '''
    Several conversations will be a list of lists containing a dict for each user. 
    '''

    if round == 1:
        return [
            [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_template.format(claim=ex['text'])},
            ]
            for ex in examples
        ]
    if round == 2 and not history:
           return [
               [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_template.format(claim=ex['claim'], label_sender=ex['label_sender'], explanation_sender=ex['explanation_sender'])},
               ]
               for ex in examples
           ]
    if round == 2 and history:
           return [
               [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_template.format(claim=ex['claim'], label_sender=ex['label_sender'], explanation_sender=ex['explanation_sender'], label_receiver=ex['label_receiver'], explanation_receiver = ex['explanation_receiver'])},
               ]
               for ex in examples
           ]
    
    raise ValueError('Not valid round or history')

def setup_logging(model_name: str, dataset: str, round: Literal[1, 2], logger_name: str = 'inference_logger', level = logging.DEBUG) -> logging.Logger:

    log_path = pathlib.Path(f'inference_logs/{'first' if round == 1 else 'second'}_round/')
    log_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_path / f'{model_name}_{dataset}_{timestamp}.log'

    # Create logger 
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    if logger.hasHandlers():
        logger.handlers.clear()


    # File handler for INFO 
    file_handler = logging.FileHandler(log_file, mode = 'w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(file_formatter)

    logger.addHandler(file_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger

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
            
