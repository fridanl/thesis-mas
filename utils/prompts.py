import json
from enum import Enum
from pydantic import BaseModel
from typing import List, Dict


class LabelSarc(str, Enum):
    sarcastic = 'sarcastic'
    literal = 'literal'

class OutputSarc(BaseModel):
    label: LabelSarc
    explanation: str
    confidence: int

class OutputSarcRound2(BaseModel):
    label: LabelSarc
    confidence: int

SARCASTIC_SCHEMA = OutputSarc.model_json_schema()
SARCASTIC_SCHEMA_ROUND2 = OutputSarcRound2.model_json_schema()


SYSTEM_JSON_GUIDED_R1 = (
    "Return ONLY a minified JSON object that conforms to this schema:\n"
    f"{json.dumps(SARCASTIC_SCHEMA, ensure_ascii=False)}\n\n"
    "Rules:\n"
    "- Keys: label, explanation, confidence.\n"
    "- label MUST be one of: 'sarcastic', 'literal'.\n"
    "- explanation MUST be 30 words or fewer.\n"
    "- confidence MUST be an integer between 0 and 100 (inclusive), no percent sign, no text.\n"
    "- Output JSON only. No prose, no Markdown, no extra text."
)

SYSTEM_JSON_GUIDED_R2 = (
    "Return ONLY a minified JSON object that conforms to this schema:\n"
    f"{json.dumps(SARCASTIC_SCHEMA_ROUND2, ensure_ascii=False)}\n\n"
    "Rules:\n"
    "- Keys: label, confidence.\n"
    "- label MUST be one of: 'sarcastic', 'literal'.\n"
    "- confidence MUST be an integer between 0 and 100 (inclusive), no percent sign, no text.\n"
    "- Output JSON only. No prose, no Markdown, no extra text."
)

USER_R1 = ( "Task: Is this claim sarcastic or literal?\n"
            'Claim: "{claim}"\n'
            "Output format:\n"
            "Label: sarcastic/literal\n"
            "Explanation: Describe your reasoning.\n"
            "Confidence (0-100): How confident are you about your answer?"
    )

USER_R2 = ( "Task: Is this claim sarcastic or literal?\n"
            'Claim: "{claim}"\n'
            "A peer of yours think the claim is {label_sender}, with the following explanation: '{explanation_sender}' \n"
            "Output format:\n"
            "Label: sarcastic/literal\n"
            "Confidence (0-100): How confident are you about your answer?"
    )