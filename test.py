import argparse, pathlib
from dotenv import load_dotenv
from utils.prompt_registry import get_prompt_spec
from utils import io
import time 


# Hugginface token 
home_env = pathlib.Path.home() / ".env"
if home_env.exists():
    load_dotenv(home_env, override=False)

def main(args):
    spec = get_prompt_spec(dataset=args.dataset, round=args.round, history=args.history)

    i = 0 
    for batch in io.load_claims_batches(path = args.dataset_path, start = args.idx_start, batch_size = args.batch_size, limit=args.limit):
        print(f'Batch number {i}')
        i += 1 
        print(batch)
        print(len(batch))

        conversations = io.build_conversations(
            examples=batch, 
            system_prompt=spec.system, 
            user_template=spec.user_template,
            history=spec.history,
            round=spec.round)
        
        


if __name__ == '__main__':
    # t0 = time.perf_counter()
    ap = argparse.ArgumentParser(description='Run offline inference on dataset (one example per line)')
    # ap.add_argument('--model_name',
    #                 help = 'Short name of model from configs/models.yaml')
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
    # ap.add_argument('--decoding_cfg', 
    #                 help='Path to YAML file with sampling params and guided decoding toggle',
    #                 default='configs/decoding.yaml')
    # ap.add_argument('--outdir',
    #                 help='Directory to write results',
    #                 default='/results/'),
    # ap.add_argument('--system', 
    #                 help = 'System prompt string',
    #                 default=SYSTEM_JSON_GUIDED_R1)
    # ap.add_argument('--user', 
    #                 help= 'User prompt string',
    #                 default=USER_R1)
    ap.add_argument('--batch_size',
                    help='Batch size to process dataset in',
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
