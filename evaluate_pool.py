from model import build_model
from dataset import build_dataset
from utils import seed_everything

import torch
import hydra
import time
import os
from torch.utils.data import Subset, DataLoader
import logging
import json
from omegaconf import OmegaConf


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(args):
    # Print Args for Identification
    logging.info('Using config: \n%s', OmegaConf.to_yaml(args))

    # Create directories necessary for output and model savings
    os.makedirs(args.path.output_dir, exist_ok=True)

    # Enable Reproducability
    seed_everything(args.random_seed)

    # Data and algorithm specific methods
    logging.info(f">>> Initialize Dataset {args.dataset}.")
    dataset, _, _, test_dataset, ds_info = build_dataset(args)
    with open(os.path.join(args.path.final_pool_dir,'indices.json'), 'r') as f:
        labeled_indices = json.load(f)
    labeled_dset = Subset(dataset, labeled_indices)

    # Initialize Model
    logging.info(f">>> Initialize Model {args.model.name}.")
    model, optimizer, lr_scheduler, train, train_criterion, eval, eval_criterion = build_model(args, ds_info=ds_info)
    random_weights_path = os.path.join(args.path.model_dir, 'random', args.model.name+"_"+str(args.random_seed)+".pth")
    pretrained_weights_path = os.path.join(args.path.model_dir, 'pretrained', args.model.name+"_"+str(args.random_seed)+".pth")
    if args.pretrain.use and os.path.exists(pretrained_weights_path):
        model.load_state_dict(torch.load(pretrained_weights_path))
        logging.info(f"Loading pretrained weights from {pretrained_weights_path}!")
    elif os.path.exists(random_weights_path):
        model.load_state_dict(torch.load(random_weights_path))
        logging.info(f"Loading deterministically initialized weights from {random_weights_path}!")
    else:
        logging.info(f"ERROR! No weights available! Model weights are randomly initialized. Be aware of random differences between different GPUs!")
            

    # Train Model
    logging.info("[Setup] Training Model.")
    labeled_train_loader = DataLoader(labeled_dset, batch_size=args.model.train_batch_size, drop_last=(args.model.train_batch_size < len(labeled_indices)), shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.model.test_batch_size)
    
    train_history = []
    t1 = time.time()
    for i_epoch in range(args.model.n_epochs):
        train_stats = train(labeled_trainloader=labeled_train_loader, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, criterion=train_criterion, device=args.device)
        train_history.append(train_stats)
        lr_scheduler.step()
        logging.info(f"[Epoch {i_epoch}][Train Stats] {train_stats}")
            
    t2 = time.time()
    training_time = t2 - t1
    logging.info(f" Training took {round(training_time, 2)} seconds.")

    eval_stats = eval(model, test_loader, eval_criterion)
    logging.info(f"[Eval Stats][Final]  {eval_stats}")

    history = {
        'train_history':train_history,
        'eval_stats':eval_stats,
        'training_time':training_time
    }

    logging.info(f"Experiment finished. Dumping results.")

    #Export results
    with open(os.path.join(args.path.output_dir, 'results.json'), 'w') as f:
        json.dump(history, f)


if __name__ == '__main__':
    main()