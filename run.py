import argparse, pathlib
from dotenv import load_dotenv
from utils.prompt_registry import get_prompt_spec
from utils import io
from utils.model_helpers import *
import time 
import logging 
import pathlib


# Hugginface token 
home_env = pathlib.Path.home() / '.env'
if home_env.exists():
    load_dotenv(home_env, override=False)

def main(args):
    script_start_time = time.time()

    model_name = args.model_name

    if args.no_logging:
        # Disable all logging
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
        logger.setLevel(logging.CRITICAL + 1)
    else:
        logger = io.setup_logging(model_name=model_name, dataset=args.dataset, round=args.round)
    
    if args.slurm_output:
        logger.info(f'SLURM output file: {args.slurm_output}')
    
    # Path for writing results  
    pre = 'first' if args.round == 1 else 'second'
    outdir = pathlib.Path(args.outdir) / pathlib.Path(pre)
    outdir.mkdir(parents = True, exist_ok = True)
    csv_path_valid = outdir / f'{model_name}-{args.dataset}.csv'

    logger.info('='*50)
    logger.info('Starting inference pipeline')
    logger.info('='*50)
    logger.info('Arguments: %s', vars(args))
    logger.info(f'Results will be written to: {csv_path_valid}')
    
    spec = get_prompt_spec(dataset=args.dataset, round=args.round, history=args.history)

    model_config = get_model_config(pathlib.Path(args.models_config_path), model_name)

    repo_id = model_config.pop('repo_id')

    logger.info('Ensuring local model availability...')
    local_model_path = ensure_local_model(repo_id=repo_id)
    model_config['repo_id'] = str(local_model_path)
    logger.info(f'Location of model: {str(local_model_path)}')
    
    llm = init_llm(model_cfg=model_config)

    shared_decoding_config = {'n': args.repetition, 'max_tokens': 254, 'use_guided_json': True}
    logger.info(f'{'#'*10} Shared decoding parameters {'#'*10}')
    logger.info(f'Repetitions (no. outputs from one input): {shared_decoding_config['n']} \n Max tokens: {shared_decoding_config['max_tokens']}')

    # Specify sampling configs, use default use available + global, else use specified in models.yaml file. 
    if model_config['has_default_sampling_params']:
        logger.info('The model has been set to have default sampling params.')
        sampling = init_sampling_params(decoding_cfg=shared_decoding_config, default=llm.get_default_sampling_params(), schema=spec.output_json)
       
    else:
        logger.info('Using sampling params as specified in models.yaml.')
        sampling_config = {**shared_decoding_config, **model_config['sampling']} # merge the global configs (decoding) and the model-specific specified in models.yaml
        sampling = init_sampling_params(sampling_config, default = None, schema=spec.output_json)
    
    logger.info(f'{'#'*10} Sampling parameters {'#'*10} \n {sampling}')

    total_inference_time = 0 
    no_rows = 0 
    batch_count = 0 
    total_failed = 0 

    n_repetitions = args.repetition # this is going to be 1
    for batch in io.load_claims_batches(path = args.dataset_path, round=pre, start = args.idx_start, batch_size = args.batch_size,    limit=args.limit):
        batch_count += 1 

        conversations = io.build_conversations(
            examples=batch, 
            system_prompt=spec.system, 
            user_template=spec.user_template,
            history=spec.history,
            round=spec.round)

        logger.debug(f"!!!!!!!!!!!!!!!!!!!!!!! see if conversations are fine!!!!!!!!!!!!!!!!!!!!!!!! {conversations}")

        start_time = time.time()
        break
        raw_outputs, parsed = run_inference(llm, conversations=conversations, sampling=sampling, output_model=spec.output_model)
        inference_time = time.time() - start_time
        total_inference_time += inference_time

        rows = []
        failed_examples = []
        # Loop over each claim/row in dataset 
        for row_idx, data in enumerate(batch):
            # Slice the outputs for specific row 

            start_idx = row_idx * n_repetitions
            end_idx = start_idx + n_repetitions

            example_texts = raw_outputs[start_idx:end_idx]
            example_parsed = parsed[start_idx:end_idx]
            if pre == "first":
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
                        'valid_json': True
                    })
                    else:
                        failed_examples.append({
                            'id': data['id'],
                            'claim': data['text'],
                            'repetition': rep_idx,
                            'raw_text': raw})
            else: # pre = second
                # Loop over an check if valid outputs 
                for rep_idx, (raw, p) in enumerate(zip(example_texts, example_parsed)):
                    if p is not None:
                        rows.append({
                        'id': data['id'], 
                        'claim': data['text'], 
                        'model_sender': data['model_sender'],
                        'model_receiver': data['model_receiver'],
                        #'repetition': rep_idx,                      # we dont need repetition
                        'label_receiver_now': p['label'],
                        'label_sender_before': data['label_sender'],
                        'label_receiver_before': data['label_receiver'],
                        'valid_json': True
                    })
                    else:
                        failed_examples.append({
                            'id': data['id'],
                            'claim': data['text'],
                            'model_sender': data['model_sender'],
                            'model_receiver': data['model_receiver'],
                            'repetition': rep_idx,
                            'raw_text': raw})

        if rows:
            io.write_csv(rows, csv_path_valid, list(rows[0].keys())) 
            no_rows += len(rows)
            logger.info(f'Writing {len(rows)} valid results to CSV')

        if failed_examples:
            total_failed += len(failed_examples)
            csv_path_failed = outdir /f'{model_name}-{spec.dataset}-failed.csv'
            io.write_csv(failed_examples, csv_path_failed, list(failed_examples[0].keys()))

        
    script_end_time = time.time()
    total_script_time = script_end_time - script_start_time
    overhead_time = total_script_time - total_inference_time

    logger.info("="*50)
    logger.info("Inference pipeline completed")
    logger.info("="*50)
    logger.info("Total batches processed: %d", batch_count)
    logger.info("Total successful results: %d", no_rows)
    logger.info("Total failed examples: %d", total_failed)
    logger.info("Output directory: %s", outdir)
    logger.info("="*50)
    logger.info("Timing summary")
    logger.info(f"Total script execution time: {io.format_time(total_script_time)} ({total_script_time:.2f}s)")
    logger.info(f"Total inference time: {io.format_time(total_inference_time)} ({total_inference_time:.2f}s)")
    logger.info(f"Overhead time (I/O, parsing, etc.): {io.format_time(overhead_time)} ({overhead_time:.2f}s)")
    logger.info(f"Inference percentage: {(total_inference_time/total_script_time)*100:.1f}%")
    if batch_count > 0:
        logger.info(f"Average time per batch: {total_script_time/batch_count:.2f}s")
    if no_rows > 0:
        logger.info(f"Average time per result: {total_script_time/no_rows:.2f}s")
    logger.info("="*50)

        

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
    ap.add_argument('--slurm_output', 
                    help='SLURM output file path', 
                    default=None)
    ap.add_argument('--history',
                    help = 'Include an agents own answer.',
                    action='store_true') # Default is False 
    ap.add_argument('--models_config_path',
                    help='Path to YAML file with model parameters.',
                    default='configs/models.yaml')
    ap.add_argument('--outdir',
                     help='Directory to write results',
                     default='results')
    ap.add_argument('--batch_size',
                    help='Number of examples to run',
                    type = int,
                    default=256)
    ap.add_argument('--no_logging',
                    action='store_true', # Default is false, i.e. logging is default
                    help='Disable all logging (no log file will be created)')
    ap.add_argument('-limit', 
                    help='Limit number of examples for inference',
                    type=int)
    ap.add_argument('-idx_start',
                    help='Idx of row to start from in dataset',
                    type=int,
                    default=0)
    
    args = ap.parse_args()
    main(args)
