import os
import subprocess
import tempfile as tmp

import pandas as pd
import torch

from typing import Dict, List

def smiles_to_temp_csv(smiles_list: List[str]) -> str:
    with tmp.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as tmp_file:
        tmp_file.write("SMILES\n")
        for smiles in smiles_list:
            tmp_file.write(f"{smiles}\n")
        return tmp_file.name
    
def _use_chemprop(args: List[str]):  # TODO optionally add code to handle error w/ capture_output=True and subprocess.CalledProcessError
    cmd = ['chemprop'] + args
    subprocess.run(cmd,capture_output=False, text=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def get_args_after_flag(args, target_flag):
    try:
        index = args.index(target_flag) + 1
    except ValueError:
        return []  

    models = []
    while index < len(args) and not args[index].startswith('--'):
        if os.path.isdir(args[index]):
            #models = []
            for root, _, files in os.walk(args[index]):
                for f in files:
                    if f.endswith('.pt'):
                        path = os.path.join(root, f)
                        models.append(path)
            return models
        elif args[index].endwith('.pt'):
            models.append(args[index])
        index += 1
    return models



class CustomChemprop:
    '''
    Class for custom chemprop models with identity crisis.
    
    '''

    def __init__(self, args, target_properties=None, desirability_configs_list=None):
        self.args = args

        model_paths = get_args_after_flag(self.args, '--model-paths')
        model_names = []
        for p in model_paths:
            model = torch.load(p, weights_only=False)
            model_names.append('Chemprop_' + model['output_columns'][0])
        # TODO find a way to safely unload the model
        self.models = model_names
        self.num_models = len(self.models)
        target_properties.remove('Chemprop')
        target_properties.extend(model_name for model_name in self.models)
        if desirability_configs_list:
            for cfg in desirability_configs_list:
                        if cfg['property'] == 'Chemprop':
                            cp_dict = cfg.copy()
        if target_properties:                    
            for prop in target_properties:
                        if prop.startswith('Chemprop_'):
                            cfg = cp_dict.copy()
                            cfg['property'] = prop
                            desirability_configs_list.append(cfg)


    def infere_chemprop(self, smiles: List[str], ret_df: bool=False) -> Dict[str, torch.tensor]:
        tmp_path = smiles_to_temp_csv(smiles)
        args = ['predict', '--test-path', tmp_path, '--num-workers', '12'] + self.args
        _use_chemprop(args)

        if self.num_models > 1:
            pred_path = tmp_path.split('.')[0] + '_preds_individual.csv'
        else:
            pred_path = tmp_path.split('.')[0] + '_preds.csv'

        df = pd.read_csv(pred_path)
        os.remove(tmp_path)
        os.remove(pred_path)

        if '--smiles-columns' in args:
            idx = args.index('--smiles-columns')
            smiles_col = args[idx + 1]
        else:
            smiles_col = 'SMILES'
        if ret_df:
            return df
        else:
            cols = [col for col in df.columns if col != smiles_col]
            cp_scores = {model_name : torch.tensor(df[col].values) for model_name, col in zip(self.models, cols)}
            return cp_scores



    
   