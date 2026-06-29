# The Impact of Reward Scalarization and Weight Scheduling on Optimization Dynamics in Multi-Objective Molecular Design

This repository contains the codebase for our study conducted in  "The Impact of Reward Scalarization and Weight Scheduling on Optimization Dynamics in Multi-Objective Molecular Design".
We provide the tools to analyze, stabilize, and optimize the "reward landscape" of molecular generators, specifically focusing on the interplay between scalarization functions and their weights.

This framework supports the full generative lifecycle, from pre-training custom base models to RL fine-tuning for specific molecular properties. Additionally, it includes streamlined utilities for sampling and property scoring of generated chemical libraries.

The code is built to be modular, allowing you to plug in your own objective and desirability functions, scalarization methods or weight assignment schemes, or use one the provided onces:

- Objective Functions:
    - Molecular Weight
    - logP
    - Hydrogen Bond Donor
    - Hydrogen Bond Akzeptor
    - Topological Surface Area
    - Number of Rotational Bonds
    - Number of Aromatic Rings
    - Number of Aliphatic Rings
    - Ratio of sp3 hybridized carbon atoms
    - BertzCT
    - Quantitative Estimate of Drug-likeliness
    - Synthetic Accessability

- Desirability Functions:
    - Identity Function (returns the input)
    - Step Function
    - Normed Gaussian
    - Sigmoid Function
    - Linear Ramp

- Scalarization Methods:
    - Sum
    - Product
    - Chebyshev
    - Smooth Chebyshev
    - Minkowski-Distance

- Weight Assignment Schemes:
    - Static
    - Randomized (Weights sampled from Dirichlet Distribution)
    - Cyclic 
    - EMA based Methods:
        - Performance
        - Improvement Rate
        - Variance


The codebase is structured as follows:

```
├── config/         # Contains structural alerts and optional configs
├── data/           # Processed data used for pre-training of the Basemodel
├── dynamicslmpo/   # Codebase
    ├── agent.py                # Contains RL Agent Setup
    ├── constraints.py          
    ├── custom_chemprop.py      # Implementation for custom chemprop models as scorer
    ├── datasets.py             
    ├── filters.py              
    ├── losses.py               
    ├── model_definition.py     
    ├── mpo.py                  # Contains Weighting Strategies
    ├── RL_trainer.py           
    ├── sampler.py              
    ├── scoring.py              
    ├── supervised_trainer.py   
    ├── utils.py                
    ├── vocabulary.py           
├── models/         # Trained Basemodel
├── vocabulary/     # Vocabulary for Basemodel
├── environment.yml # Dependency list
└── README.md
```

## Installation

```bash
git clone https://github.com/palominohernandez/ExploRL25.git
cd dynamicalmpo
conda env create -f environment.yml
conda activate dynamicalmpo
```


## Usage 

The following serves as a quick guide to use the different core functionalities.

### Train a Supervised Model Example

python main.py --mode train_supervised \
    --data_file path/to/dataset.csv --vocab_path vocabulary/vocab.json \
    --supervised_model_path models/name_of_model.pth \
    --results_dir results/supervised_training \
    --supervised_checkpoint_dir checkpoints/supervised \
    --epochs 3 --batch_size 128 \
    --learning_rate 1e-3 \
    --val_size 0.1 --subset_fraction 0.05


### Sampling From Model Example

When sampling the model architecture has to be implemented prior to usage.

python main.py --mode sample --load_agent_model_path /path/to/model.pth \
    --vocab_path /path/to/vocab.json \
    --output_samples_file /output/path/for/generated_samples.smi \
    --num_samples 1000 --sample_batch_size 128 \
    --temperature 0.5 

### Scoring Example
python main.py \
    --mode scoring \
    --input_smiles_file /path/to/sampled/molecules.smi \
    --output_scores_file /path/to/output/file.csv \
    --results_dir /path/to/raw/score/output/ \
    --vocab_path vocabulary/chembl_vocab.json \
    --target_properties QED SA LogP MW HBD HBA \
    --target_values 0.7 0.8 3.0 400.0 3.0 6.0 \
    --mpo_strategy static \
    --mpo_static_weights 0.166 0.166 0.166 0.166 0.166 0.166 \
    --mpo_combination sum \
    --enable_structural_alerts \
    --structural_alerts_path config/structural_alerts.txt \
    --enable_property_limits \
    --property_limits_config '[{"property": "MW", "op": "<=", "value": 500}, {"property": "LogP", "op": "<=", "value": 5}, {"property": "HBD", "op": "<=", "value": 5}, {"property": "HBA", "op": "<=", "value": 10}]' \
    --disable_diversity_filter

### RL Tuning Example 
export DESIRABILITY_JSON='[{"property": "QED", "type": "identity", "params": {}}, {"property": "SA", "type": "linear_ramp", "params": {"low": 6.0, "high": 1.0, "target_value": 1.0}}]'                 

python main.py --mode reinforce \
    --load_agent_model_path models/smiles_rnn_supervised.pth \
    --prior_model_path models/smiles_rnn_supervised.pth \
    --vocab_path vocabulary/chembl_vocab.json \
    --results_dir results/rl_qed_sa \
    --rl_checkpoint_dir checkpoints/rl_qed_sa \
    --target_properties QED SA \
    --desirability_configs "${DESIRABILITY_JSON}" \
    --target_values 0.9 0.95 \
    --loss_function reinvent \
    --reinvent_sigma 60.0 \
    --reinvent_kl_beta 0.5 \
    --reinvent_entropy_beta 0.5 \
    --mpo_strategy dynamic \
    --mpo_perf_gap_beta 0.05 \
    --rl_epochs 500 \
    --rl_batch_size 64 \
    --rl_lr 5e-5 \
    --enable_structural_alerts \
    --structural_alerts_path config/structural_alerts.txt \
    --diversity_filter_strategy IdenticalMurcko \
    --diversity_bucket_capacity 25 \
    --diversity_filter_threshold 0.5















## Cite this work

Preprint available at [ChemRxiv](https://chemrxiv.org/doi/10.26434/chemrxiv-2025-pmnmb)

