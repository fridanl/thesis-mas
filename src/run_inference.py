from dotenv import load_dotenv
import argparse, pathlib
from utils.data import load_claims_text, load_claims_batches
from utils.prompts import build_conversations, SYSTEM_JSON_GUIDED_R1, USER_R1
from utils.models import load_yaml, init_llm, init_sampling_params, ensure_local_model
from utils.runner import run_inference
from utils.io import write_csv, write_jsonl
from utils.prompts import OutputSarc
import time

home_env = pathlib.Path.home() / ".env"
if home_env.exists():
    load_dotenv(home_env, override=False)

def main(args):
    # load model configs
    default_cfg = load_yaml('configs/default-model.yaml')
    profiles_root = load_yaml('configs/models.yaml')
    profiles = profiles_root.get('profiles', {})

    model_name = args.model_name
    if model_name not in profiles:
        raise SystemExit(f"Unknown --model_name '{model_name}'. Available: {', '.join(profiles.keys())}")
    
    # Configs that are passed when initialising model
    model_cfg = {**default_cfg, **profiles[model_name]}

    # making sure to run local model, and download if not there 
    repo_id = model_cfg['model']
    print(repo_id)
    local_path = ensure_local_model(repo_id=repo_id)
    print(local_path)
    model_cfg['model'] = str(local_path)


    # init model
    llm = init_llm(model_cfg=model_cfg)
    
    # get sampling params
    decoding_cfg = load_yaml(args.decoding_cfg)

    if model_cfg['has_default_sampling_params']:        
        # if sampling params specified in huggingface repo
        sampling = init_sampling_params(decoding_cfg, default = llm.get_default_sampling_params())
    else:
        # params we have specified
        decoding_cfg = {**decoding_cfg, **model_cfg['sampling']}
        sampling = init_sampling_params(decoding_cfg, default = None)
    
    # print params to output
    print('###### SAMPLING PARAMS ######')
    print(sampling)

    # write results 
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents = True, exist_ok = True)
    
    jsonl_path = outdir / f'first-{model_name}.jsonl'
    csv_path = outdir / f'first-{model_name}.csv'

    no_rows = 0
    
    for batch in load_claims_batches(path = args.dataset_path, start = args.idx_start, batch_size = args.batch_size, limit=args.limit):
         # build the prompts
        conversations = build_conversations(
            examples=batch, 
            system_prompt=args.system, 
            user_template=args.user)

        for i in range(args.repetition):
            # run inference 
            texts, parsed = run_inference(llm, conversations=conversations, sampling=sampling, json_format=OutputSarc)
            valid_json = []

            rows = []
            for ex, t, p in zip(batch, texts, parsed):
                valid_json = False 
                if p is not None:
                    valid_json = True
                    rows.append({'id': ex['id'], 'claim': ex['text'], 'model': model_name, 'repetition': i, 'label': p['label'], 'explanation': p['explanation'], 'confidence': p['confidence'], 'valid_json': valid_json, 'raw_text': t})

                else:
                    rows.append({'id': ex['id'], 'claim': ex['text'], 'model': model_name, 'repetition': i, 'label': None, 'explanation': None, 'confidence': None, 'valid_json': valid_json, 'raw_text': t})

            no_rows += len(rows)
            write_csv(rows, csv_path, ['id', 'claim', 'model', 'repetition', 'label', 'explanation', 'confidence', 'valid_json', 'raw_text'])

    print(f'Wrote {no_rows} rows to {outdir}')

if __name__ == '__main__':
    t0 = time.perf_counter()
    try:
        ap = argparse.ArgumentParser(description='Run offline inference on dataset (one example per line)')
        ap.add_argument('--model_name',
                        help = 'Short name of model from configs/models.yaml')
        ap.add_argument('--dataset_path', 
                        help='Path to dataset', 
                        default='data/sarc/sarcasm2.csv')
        ap.add_argument('--repetition',
                        help='Number of times a model is presented a specific claim.',
                        type=int,
                        default=1)
        ap.add_argument('--decoding_cfg', 
                        help='Path to YAML file with sampling params and guided decoding toggle',
                        default='configs/decoding.yaml')
        ap.add_argument('--outdir',
                        help='Directory to write results',
                        default='/results/'),
        ap.add_argument('--system', 
                        help = 'System prompt string',
                        default=SYSTEM_JSON_GUIDED_R1)
        ap.add_argument('--user', 
                        help= 'User prompt string',
                        default=USER_R1)
        ap.add_argument('--batch_size',
                        help='Batch size to process dataset in',
                        type = int,
                        default=256)
        ap.add_argument('-limit', 
                        help='Limit number of examples for inference',
                        type=int)
        ap.add_argument('-idx_start',
                        help='Idx of row to start from in dataset',
                        type=int,
                        default=0)

        args = ap.parse_args()

        main(args)
    finally:
        dt = time.perf_counter() - t0 
        print(f'[timing] total runtime: {dt:.3f}s')
