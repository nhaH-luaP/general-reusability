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
        p_path = os.path.join(args.path.initial_pool_dir, args.dataset+'_'+str(args.al.initial_pool_size)+'_'+str(seed+1)+'.json')
        if not os.path.exists(p_path):
            logging.info(f"Created new pool at {p_path}!")
            with open(p_path, 'w') as f:
                json.dump(initial_labeled_pool, f)
        else:
            logging.info(f"There is already a pool at {p_path}!")

        model, _, _, _, _, _, _ = build_model(args=args, ds_info=ds_info)
        m_path = os.path.join(args.path.model_dir, args.model.name+'_'+str(seed+1)+'.pth')
        if not os.path.exists(m_path):
            logging.info(f"Created new model at {m_path}!")
            state_dict = model.state_dict()
            torch.save(state_dict,m_path)
        else:
            logging.info(f"There is already a model at {m_path}!")


if __name__ == '__main__':
    main(None)