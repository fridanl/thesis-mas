# Influence Dynamics in Heterogeneous LLM-Based Multi-Agent Systems

## Abstract 

Large Language Models (LLMs) are increasingly deployed in Multi-Agent Systems (MAS) where multiple agents collaborate towards a shared solution. 
While prior work has shown that opinion dynamics in these systems are complex and can be misaligned with expectations, little focus has been paid to how these differ when agents are powered by distinct model types.
In systems where models differ in size, architecture and training data, it remains unclear how agents can influence each others' outcomes, and whether a shift to a heterogeneous setting affects influence dynamics. 
We propose a framework for measuring influence dynamics in LLM-based MAS. 
An agent is tasked with a binary classification task in isolation, and is later asked to revisit this task, but now considering another agent's opinion. 
Influence is measured as the shift in the agent's certainty in its opinion before and after interaction.
We evaluate several open-source models on three natural language processing tasks. 
Results show that interaction alone produces frequent and substantial opinion shifts, even for models that are initially certain in their opinion. 
Furthermore, influence is driven mainly be the stability of the model being influenced and less by the influential power of the peer.
Similarly, the transition from homogeneous to heterogeneous systems does not affect all models uniformly. Depending on the model, heterogeneity may either increase or decrease susceptibility. 
Overall, system composition, model types, and model roles are important factors in influence dynamics, and thus need to be considered when building robust MAS.

## Usage

### Command Line Arguments

The main Python script `run.py` accepts the following arguments:

-   `--model_name` (str, required): Name of model under evaluation. Format as specified in the `models.yaml`. 
-   `--dataset` (str, default: 'sarcasm'): Name of dataset under evaluation.
-   `--dataset_path` (str, required): Path to dataset/input.
-   `--repetition` (int, default: 1): Number of times a model is prompted per input.
-   `--round` (int, default: 1): Round of experiment. 1: solo inference, 2: pairwise interaction.
-   `--models_config_path` (str, default: configs/models.yaml): Path to YAML file with model parameters. 
-   `--outdir` (str, default: results): Directory to write results. 
-   `--batch_size` (int, default: 256): Number of claims per inference. Note, this is multiplied by args.repetition. 
-   `--slurm_output` (str, required): Name of SLURM output file. For reporting in inference file. 
-   `--history`: Flag to include history, i.e. an agent's previous answer, in round 2 prompts. 
-   `--no_explanation`: Flag to exclude explanations from round 2 prompts. 
-   `--no_logging`: Flag to disable logging in inference log file. 
-   `--idx_start` (int, default: 0): Index of row to start from in dataset. 
-   `-limit` (int): Limit number of examples for evaluation. Note, this is on a claim level. 

### Example Script 

The script is used for both the first and second round. 

#### Example: Round 1 
```BASH
uv run run.py 
    --model_name llama-3.1-8b \
    --dataset sarcasm \
    --dataset_path data/sarc/sarcasm.csv \
    --repetition 10 \
    --round 1 \
    --slurm_output "${SLURM_OUTPUT_FILE}"
```

Use `python3 run.py` if you are not using `uv`.

#### Example: Round 2 
```BASH
uv run run.py 
    --model_name llama-3.1-8b
    --dataset sarcasm \
    --dataset_path /subsampled_input_round2/sarcasm/llama-3.1-8b_disagree.csv \ # Path to constructed opinion sets
    --repetition 1 \  #1 repetition, since the opinion sets have 10 repetitions.
    --round 2 \
    --history \
    --slurm_output "${SLURM_OUTPUT_FILE}"
```

### Construction of opinion sets: Second round input 

We perform a sequence of steps, described below, in order to prepare the results of first round to perform experiments on the interaction between agents. 

The preprocessing consists of two main steps; pairwise matching (with accompanied match type) and subsampling.

We define the model pairs under evaluation to be intra-family and all combinations of the large models. 

Based on the specific model pairs and the different match types, we split the input for round 2 into two files for each receiver model. Namely, for agreeing and disagreeing cases. 
Additionally, self-interaction (i.e. homogeneous system) opinions sets can be created too. 

This all happens in `src/make_second_round_input.py`.

```BASH
uv run -m src.make_second_round_input 
    --dataset sarcasm

# Or for self-interaction 
uv run -m src.make_second_round_input 
    --dataset sarcasm \
    --self_interaction
```

This will create one input file for every receiver model, paired with all specified peer models.

**Subsampling**

If one wishes to subsample the input for round 2, one can do so with the script: `src/make_subsample.py`. 

```BASH
uv run src/make_subsample.py 
    --suffix disagree \
    --cap 7000 \ # Note: cap specifies the # claims, therefore the size of the dataset will become cap*repetition.
    --dataset sarcasm \

# Or for self-interaction agreeing
uv run src/make_subsample.py 
    --glob_pattern *_self_interaction_agree.csv \
    --cap 1000 \
    --dataset sarcasm
```
## Evaluation

When all runs are done, the results can be computed by running the `src/results.py` script. 
This will compute influence scores for every model and dataset, and create files of three aggregation levels: on a claim level, on a match type level, and on a model level. 

```BASH
uv run src/results.py 
    --dataset sarcasm \
    --experiment main
```

## Results reported in paper

All tables and figures reported in the paper is created in the notebook: `src/main_results.ipynb`.

## Input Files and Raw Outputs

All raw results are available at our [figshare repository](https://figshare.com/s/0663638a98e0efc47105).

The data repository contains the following data: 
- `first.zip`: Output files of first round.
- `second.zip`: Output files of second round (interaction).
- `self.zip`: Output files of second round homogeneous interactions.
- `input_round2.zip`: Input files for second round interaction. File name specify the receiver model. The files contain all available interactions. 
- `subsampled_input_round2.zip`: Same as input_round2.zip, but with downsampled instances.
- `swap.zip`: Output files from swap experiment. 
- `temperature.zip`: Input and output files from temperature experiment.
- `no_history.zip`: Output files from memoryless receiver experiment.
- `no_explanation.zip`: Output files from no explanations experiment. 


## Environment

This project is configured to run on NVIDIA GPUs and has been developed and tested using CUDA 12.6. If you plan to run this on a different CUDA version then you might need to adjust the environment.

Depending on your preferred method the environment can be set up using either uv or pip.

**If you are using `uv`:**

```BASH
uv venv 
source .venv/bin/activate # Linux/macOS 

uv sync # to install dependencies
```
**If you are using pip:**

We also provide a requirements.txt file pip if preferred. Simply run:
```BASH
pip install -r requirements.txt
```


## Data

The data folder contains data from the following sources:
- Sarcasm dataset:  [SocKet Repository](https://github.com/minjechoi/SOCKET).
- Sentiment dataset:  [SocKet Repository](https://github.com/minjechoi/SOCKET).
- CommonsenseQA: [HuggingFace Datasets](https://huggingface.co/datasets/chiayewken/commonsense-qa-2).
