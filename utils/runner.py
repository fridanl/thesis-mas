import pathlib, yaml, json 
from vllm import LLM
from vllm.sampling_params import SamplingParams, GuidedDecodingParams
import os, subprocess
from typing import List, Dict, Any, Tuple, TypedDict, Sequence, Type, Optional
from pydantic import ValidationError, BaseModel

class ChatCompletionMessageParam(TypedDict):
    role: str
    content: str

# A conversation is a list of a dict with keys; role, content 
Conversation = List[ChatCompletionMessageParam]
Conversations = Sequence[Conversation]

def load_yaml(path: str) -> dict:
    return yaml.safe_load(pathlib.Path(path).read_text())

def _sanitize_repo_id(repo_id: str) -> str:
    # HF hub’s cache uses "models--org--name" style
    return "models--" + repo_id.replace("/", "--")

def ensure_local_model(
    repo_id: str
    ) -> pathlib.Path:
    """
    Ensure a model repo is present locally and return its path.
    - repo_id: e.g. "meta-llama/Meta-Llama-3-8B-Instruct"
    """
    home_env = pathlib.Path.home()
    local_dir = pathlib.Path(f'{home_env}/.cache/huggingface/hub/{_sanitize_repo_id(repo_id)}')
    san = _sanitize_repo_id(repo_id)
    print(san)
    # If the directory already exists and is non-empty, assume it’s usable
    if local_dir.exists() and any(local_dir.iterdir()):
        return local_dir
    

    cmd = [
        "hf", "download", repo_id,
        "--local-dir", str(local_dir),
    ]

    subprocess.run(cmd, check=True)
    print('hiiiii')
    return local_dir


def init_llm(model_cfg: dict) -> LLM:
    return LLM(
        model=model_cfg['model'],
        quantization=model_cfg['quantization'],
        seed=model_cfg['seed'],
        dtype=model_cfg.get('dtype', None),
    )


def init_sampling_params(
    decoding_cfg: dict, 
    default: SamplingParams, 
    Schema) -> SamplingParams:
    """
    Function to initialise sampling params. 
    If default: then there are specified sampling params in HuggingFace repo, and we add the other attributes. 
    Otherwise we initialise samplingparams with the specified sampling params. 
    """
    guided = GuidedDecodingParams(json = Schema) if decoding_cfg['use_guided_json'] else None

    # If there are default sampling params in huggingface, use all defaults and only write over default params that are specified in decoding_cfg. 
    if default is not None:
        default.guided_decoding = guided
        for k in decoding_cfg.keys():
            if hasattr(default, k):
                setattr(default, k, decoding_cfg[k])
        return default
    else: # no default params
        allowed_keys = {
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "repetition_penalty",
            "presence_penalty",
            "frequency_penalty",
            "n",
            "max_tokens",
            "seed",
            "stop",
            "stop_token_ids",
        }
        kwargs: Dict[str, Any] = {}
        for k in allowed_keys:
            if k in decoding_cfg and decoding_cfg[k] is not None: # we have this check because there are other keys like quantization in the dict. 
                kwargs[k] = decoding_cfg[k] # Params specified in decoding (like temperature)

        return SamplingParams(
            guided_decoding = guided,
            **kwargs
            )
    

def run_inference_multi(
    llm: LLM,
    conversations: Sequence[List[ChatCompletionMessageParam]],
    sampling,
    json_format: Type[BaseModel]) -> Tuple[List[List[str]], List[List[Optional[Dict[str, Any]]]]]:

    outputs = llm.chat(messages=conversations, SamplingParams)

    texts_per_item: List[List[str]] = []
    parsed_per_item: List[List[Optional[Dict[str, Any]]]] = []

    for req_out in outputs:
        for out in req_out.outputs:
            t = out.text
            


def run_inference(
    llm: LLM, 
    conversations: Sequence[List[ChatCompletionMessageParam]],
    sampling: SamplingParams,
    json_format) -> Tuple[List[str], List[Dict[str, Any] | None], float]:
    

    outs = llm.chat(messages=conversations, sampling_params=sampling)

    texts = [o.outputs[0].text if o.outputs else "" for o in outs]
    parsed = []

    for txt in texts:
        try:
            obj = json.loads(txt)
            parsed.append(json_format(**obj).model_dump())
        except(json.JSONDecodeError, ValidationError, KeyError, TypeError):
            parsed.append(None)

    return texts, parsed