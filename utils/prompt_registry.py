import json
from enum import Enum
from pydantic import BaseModel
from typing import List, Dict, Type, Literal
from dataclasses import dataclass

# ----- Output Models, Round Specific (w/ or wo/ explanation) ----------- 
class LabelSarc(str, Enum):
    sarcastic = 'sarcastic'
    literal = 'literal'

class OutputSarc(BaseModel):
    label: LabelSarc
    explanation: str

class OutputSarcR2(BaseModel):
    label: LabelSarc


# ----- Core prompt builders ---------
def make_system_json(schema: dict, *, round: int) -> str:
    base = (
        "Return ONLY a minified JSON object that conforms to this schema: \n"
        f"{json.dumps(schema, ensure_ascii=False)}\n"
        "Rules:\n"
        "- Output JSON only. No prose, no markdown, no extra text.\n"
    )
    if round == 1:
        base += (
            "- Keys: label, explanation. \n"
            "- explanation MUST be 30 words or fewer \n"
        )
    if round == 2:
        base += (
            "- Keys: label"
        )
    return base 

def make_user_r1(*, task_question: str) -> str:
    # Task specific question as input
    return (
        f"Task: {task_question}\n"
        'Claim: "{claim}"'
    )

def make_user_r2(*, task_question: str, history: bool) -> str:
    #! Maybe the order / wording needs to be changed here 
    base = (
        f"Task: {task_question}\n"
        'Claim: "{claim}"\n'
        'A peer of yours think the claim is "{label_sender}", with the following explanation: "{explanation_sender}" \n'
    )
    if history:
        base += (
            'You have previously said "{label_receiver}", with the following explanation: "{explanation_receiver}"'
        )

    return base

# ------ Registry types ------- 
@dataclass(frozen=True)
class DatasetTaskSpec:
    '''
    Defines dataset-specific task wording and output models.
    '''

    dataset: str 
    task_question: str 
    output_r1: Type[BaseModel]
    output_r2: Type[BaseModel]


@dataclass(frozen=True)
class PromptSpec:
    dataset: str
    round: Literal[1, 2]
    history: bool 
    system: str 
    user_template: str 
    output_model: Type[BaseModel]


# ----- Dataset registry ------
DATASETS: Dict[str, DatasetTaskSpec] = {
    'sarcasm': DatasetTaskSpec(
                dataset = 'sarcasm',
                task_question= 'Is this claim sarcastic or literal?',
                output_r1=OutputSarc,
                output_r2=OutputSarcR2
                )
    #TODO: Add more datasets
                }


def get_prompt_spec(dataset: str, round: int, history: bool) -> PromptSpec:
    key = dataset.lower()
    if key not in DATASETS:
        raise ValueError(f'Unknown dataset {dataset}. Known: {', '.join(DATASETS.keys())}')
    
    ds = DATASETS[key]

    if round == 1:
        schema = ds.output_r1.model_json_schema()
        return PromptSpec(
            dataset=key,
            round = 1,
            history=history,
            system=make_system_json(schema, round=1),
            user_template=make_user_r1(task_question=ds.task_question),
            output_model=ds.output_r1
        )
    
    if round == 2:
        schema = ds.output_r2.model_json_schema()
        return PromptSpec(
            dataset=key,
            round = 2,
            history=history,
            system=make_system_json(schema, round=2),
            user_template=make_user_r1(task_question=ds.task_question),
            output_model=ds.output_r2
        )

    raise ValueError('Round must be either 1 or 2')




# SARCASTIC_SCHEMA = OutputSarc.model_json_schema()

# SYSTEM_JSON_GUIDED_R1 = (
#     "Return ONLY a minified JSON object that conforms to this schema:\n"
#     f"{json.dumps(SARCASTIC_SCHEMA, ensure_ascii=False)}\n\n"
#     "Rules:\n"
#     "- Keys: label, explanation, confidence.\n"
#     "- label MUST be one of: 'sarcastic', 'literal'.\n"
#     "- explanation MUST be 30 words or fewer.\n"
#     "- Output JSON only. No prose, no Markdown, no extra text."
# )

# SYSTEM_JSON_GUIDED_R2 = (
#     "Return ONLY a minified JSON object that conforms to this schema:\n"
#     f"{json.dumps(SARCASTIC_SCHEMA, ensure_ascii=False)}\n\n"
#     "Rules:\n"
#     "- Keys: label, confidence.\n"
#     "- label MUST be one of: 'sarcastic', 'literal'.\n"
#     "- confidence MUST be an integer between 0 and 100 (inclusive), no percent sign, no text.\n"
#     "- Output JSON only. No prose, no Markdown, no extra text."
# )

# USER_R1 = ( "Task: Is this claim sarcastic or literal?\n"
#             'Claim: "{claim}"'
#     )

# USER_R2 = ( "Task: Is this claim sarcastic or literal?\n"
#             'Claim: "{claim}"\n'
#             "A peer of yours think the claim is {label_sender}, with the following explanation: '{explanation_sender}' \n"
#             "Output format:\n"
#             "Label: sarcastic/literal\n"
#             "Confidence (0-100): How confident are you about your answer?"
#     )