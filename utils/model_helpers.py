import pathlib, yaml, json 
from vllm import LLM
from vllm.sampling_params import SamplingParams, GuidedDecodingParams
import os, subprocess
from typing import List, Dict, Any, Tuple, TypedDict, Sequence, Type, Optional
from pydantic import ValidationError, BaseModel
import pathlib

class ChatCompletionMessageParam(TypedDict):
    role: str
    content: str

def load_yaml(path: pathlib.Path) -> dict:
    return yaml.safe_load(path.read_text())

def _sanitize_repo_id(repo_id: str) -> str:
    # HF hub’s cache uses "models--org--name" style
    return "models--" + repo_id.replace("/", "--")

def get_model_config(
        config_path: pathlib.Path, 
        model_name: str
        ) -> dict:
    
    # Load all profiles 
    profiles_root = load_yaml(config_path)
    profiles = profiles_root.get('profiles', {})

    # check if model is specified in config file
    if model_name not in profiles:
        raise SystemExit(f"Unknown --model_name '{model_name}'. Available: {', '.join(profiles.keys())}")
    
    shared_configs = {
        'seed': 0,
        'tensor_parallel_size': 1,
        'dtype': 'auto' #! here add max_model_length 
    }

    # Configs that are passed when initialising model 
    model_cfg = {**shared_configs, **profiles[model_name]}

    return model_cfg

def ensure_local_model(
    repo_id: str
    ) -> pathlib.Path:
    """
    Ensure a model repo is present locally and return its path.
    - repo_id: e.g. "meta-llama/Meta-Llama-3-8B-Instruct"
    """
    home_env = pathlib.Path.home()
    
    # Look for the format model is saved in
    san_repo_id = _sanitize_repo_id(repo_id)
    local_dir = pathlib.Path(f'{home_env}/.cache/huggingface/hub/{san_repo_id}')
    
    # If the directory already exists and is non-empty, assume it’s usable
    if local_dir.exists() and any(local_dir.iterdir()):
        print(f'Model already downloaded and located at: {local_dir}. Will use already downloaded weights.')
        return local_dir
    
    print(f'Model does not exist at {local_dir} --- Starting download of model....')
    cmd = [
        "hf", "download", repo_id,
        "--local-dir", str(local_dir),
    ]

    subprocess.run(cmd, check=True)
    if local_dir.exists() and any(local_dir.iterdir()):
        print(f'Model successfully downloaded and located at: {local_dir}.')
        return local_dir
    
    raise RuntimeError(f'Could not find model at {local_dir} after attempted download.')


def init_llm(model_cfg: dict) -> LLM:
    return LLM(
        model=model_cfg['model'],
        quantization=model_cfg['quantization'],
        seed=model_cfg['seed'],
        dtype=model_cfg.get('dtype', None),
        tensor_parallel_size = model_cfg['tensor_parallel_size']
    )


def init_sampling_params(
    decoding_cfg: dict, 
    default: Optional[SamplingParams], 
    schema) -> SamplingParams:
    """
    Initialise SamplingParams.
    - If default is provided, clone and override with values from decoding_cfg.
    - If no default, construct from decoding_cfg.
    """
    
    guided = GuidedDecodingParams(json = schema) if decoding_cfg['use_guided_json'] else None

    # Extract valid fields 
    valid_fields = SamplingParams.__annotations__.keys()
    print(f'VALID FIELDS: ------------:\n {valid_fields}')
    # Extract valid fields specified in decoding, we want to set in SamplingParams
    overrides: Dict[str, Any] = {
        k: v
        for k, v in decoding_cfg.items()
        if k in valid_fields and v is not None
    }

    print(f'Overrides: \n {overrides}')

    if default is not None:
        # copy 
        params = default.__class__(**default.__dict__)
        for k, v in overrides.items():
            print(f'Overriding params: {k}, with value: {v}')
            setattr(params, k, v)
    else:
        params = SamplingParams(**overrides)
    
    params.guided_decoding = guided 
    return params

def run_inference(
    llm: LLM, 
    conversations: Sequence[List[ChatCompletionMessageParam]],
    sampling: SamplingParams,
    output_model: Type[BaseModel]):
    

    outs = llm.chat(messages=conversations, sampling_params=sampling)

    texts = [o.outputs[0].text if o.outputs else "" for o in outs]
    parsed = []

    for txt in texts:
        try:
            obj = json.loads(txt)
            parsed.append(output_model(**obj).model_dump())
        except(json.JSONDecodeError, ValidationError, KeyError, TypeError):
            parsed.append(None)

    return texts, parsed