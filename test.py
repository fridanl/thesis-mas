import argparse, pathlib
from dotenv import load_dotenv
from utils.prompt_registry import get_prompt_spec
from utils import io
from utils.model_helpers import *
import time 
import pathlib


# Hugginface token 
home_env = pathlib.Path.home() / ".env"
if home_env.exists():
    load_dotenv(home_env, override=False)

def main(args):
    # Get prompt specs
    spec = get_prompt_spec(dataset=args.dataset, round=args.round, history=args.history)
    
    model_name = args.model_name

    # Get model configs 
    model_config = get_model_config(pathlib.Path(args.models_config_path), model_name)
    print(model_config)
    repo_id = model_config.pop('repo_id')

    # Get path of local model and download if not there 
    local_model_path = ensure_local_model(repo_id=repo_id)
    model_config['repo_id'] = str(local_model_path)
    # Init model
    llm = init_llm(model_cfg=model_config)

    print(f'Default sampling params before any changes {llm.get_default_sampling_params()}') #TODO: Check here. THis was not correct for qwen model 1.5 

    shared_decoding_config = {'n': args.repetition, 'max_tokens': 254, 'use_guided_json': True}

    # Specify sampling configs, use default use available + global, else use specified in models.yaml file. 
    if model_config['has_default_sampling_params']:
        sampling = init_sampling_params(decoding_cfg=shared_decoding_config, default=llm.get_default_sampling_params(), schema=spec.output_json)
    else:
        sampling_config = {**shared_decoding_config, **model_config['sampling']} # merge the global configs (decoding) and the model-specific specified in models.yaml
        sampling = init_sampling_params(sampling_config, default = None, schema=spec.output_json)

    print('###### SAMPLING PARAMS ######')
    print(sampling)
    print('#'*30)
    

    # write results 
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents = True, exist_ok = True)
    csv_path_valid = pathlib.Path(outdir / f'first-{model_name}-{spec.dataset}.csv')

    no_rows = 0 
    for batch in io.load_claims_batches(path = args.dataset_path, start = args.idx_start, batch_size = args.batch_size, limit=args.limit):
        conversations = io.build_conversations(
            examples=batch, 
            system_prompt=spec.system, 
            user_template=spec.user_template,
            history=spec.history,
            round=spec.round)

        raw_outputs, parsed = run_inference(llm, conversations=conversations, sampling=sampling, output_model=spec.output_model)
       
        n_repetitions = args.repetition

        rows = []
        failed_examples = []  

        # Loop over each claim/row in dataset 
        for row_idx, data in enumerate(batch):
            # Slice the outputs for specific row 
            start_idx = row_idx * n_repetitions
            end_idx = start_idx + n_repetitions
    
            example_texts = raw_outputs[start_idx:end_idx]
            example_parsed = parsed[start_idx:end_idx]

            # Loop over an check if valid outputs 
            for rep_idx, (raw, p) in enumerate(zip(example_texts, example_parsed)):
                if p is not None:
                    rows.append({
                    'id': data['id'], 
                    'claim': data['text'], 
                    'model': model_name,
                    'repetition': rep_idx,
                    'label': p['label'], 
                    'explanation': p['explanation'], 
                    'valid_json': True, 
                    'raw_text': raw
                })
                else:
                    failed_examples.append({
                        'id': data['id'],
                        'claim': data['text'],
                        'repetition': rep_idx,
                        'raw_text': raw})

        # Write results to file, each batch   
        io.write_csv(rows, csv_path_valid, list(rows[0].keys())) 

        if failed_examples:
            csv_path_failed = pathlib.Path(outdir / f'first-{model_name}-{spec.dataset}-failed.csv')
            io.write_csv(rows, csv_path_failed, list(failed_examples[0].keys())) 

        no_rows += len(rows)

    print('----------------Done--------------------')
    print(f'Number of successful results: {no_rows}')


        

if __name__ == '__main__':

    ap = argparse.ArgumentParser(description='Run offline inference on dataset (one example per line)')
    ap.add_argument('--model_name',
                    help = 'Short name of model from configs/models.yaml')
    ap.add_argument('--dataset',
                    help = 'Specify name of dataset',
                    default='sarcasm')
    ap.add_argument('--dataset_path', 
                    help='Path to dataset', 
                    default='data/sarc/sarcasm.csv')
    ap.add_argument('--repetition',
                    help='Number of times a model is presented a specific claim.',
                    type=int,
                    default=1)
    ap.add_argument('--round',
                    type=int,
                    choices=[1,2],
                    default=1)
    ap.add_argument('--history',
                    action='store_true') # Default is False 
    ap.add_argument('--models_config_path',
                    help='Path to YAML file with model parameters.',
                    default='configs/models.yaml')
    # ap.add_argument('--decoding_cfg', 
    #                 help='Path to YAML file with sampling params and guided decoding toggle',
    #                 default='configs/decoding.yaml')
    ap.add_argument('--outdir',
                     help='Directory to write results',
                     default='/home/fril/thesis-mas/results/')
    ap.add_argument('--batch_size',
                    help='Number of examples to run',
                    type = int,
                    default=1)
    ap.add_argument('-limit', 
                    help='Limit number of examples for inference',
                    type=int)
    ap.add_argument('-idx_start',
                    help='Idx of row to start from in dataset',
                    type=int,
                    default=0)
    
    args = ap.parse_args()
    main(args)
