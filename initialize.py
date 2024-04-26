from dataset import build_dataset
from model import build_model

import torch
import hydra
import random
import json
import os
import logging
from omegaconf import OmegaConf

@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(args):
    # Print args
    logging.info('Using config: \n%s', OmegaConf.to_yaml(args))

    os.makedirs(args.path.initial_pool_dir, exist_ok=True)
    os.makedirs(args.path.model_dir, exist_ok=True)
    _, _, _, ds_info = build_dataset(args)
    for seed in range(args.num_seeds):
        initial_labeled_pool = random.sample(range(ds_info['n_samples']), k=args.al.initial_pool_size)
        path = os.path.join(args.path.initial_pool_dir, args.dataset+'_'+str(args.al.initial_pool_size)+'_'+str(seed+1)+'.json')
        if not os.path.exists(path):
            with open(path, 'w') as f:
                json.dump(initial_labeled_pool, f)

        model, _, _, _, _, _, _ = build_model(args=args, ds_info=ds_info)
        path = os.path.join(args.path.model_dir, args.model.name+'_'+str(seed+1)+'.pth')
        if not os.path.exists(path):
            state_dict = model.state_dict()
            torch.save(state_dict,path)


if __name__ == '__main__':
    main(None)