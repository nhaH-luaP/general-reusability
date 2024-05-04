from model import build_model
from dataset import build_dataset
from query import build_query
from utils import seed_everything

import torch
import time
import os
import hydra
import json
import random
import logging
import copy

from omegaconf import OmegaConf
from torch.utils.data import Subset, DataLoader

@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(args):
    # Print args
    logging.info('>>> Using config: \n%s', OmegaConf.to_yaml(args))

    # Create directory for pool and results
    logging.info(">>> Creating output directory at: "+str(args.path.output_dir))
    os.makedirs(args.path.output_dir, exist_ok=True)
    os.makedirs(args.path.final_pool_dir, exist_ok=True)

    # Enable reproducability
    logging.info(">>> Seed experiment with random seed {args.random_seed}.")
    seed_everything(args.random_seed + 42)
    
    # Initialize Dataset
    logging.info(f">>> Initialize Dataset {args.dataset}.")
    dataset, mixmatch_dataset, query_dataset, test_dataset, ds_info = build_dataset(args)

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

    # Initialize Pools
    initial_pool_path = os.path.join(args.path.initial_pool_dir, args.dataset+'_'+str(args.al.initial_pool_size)+'_'+str(args.random_seed)+'.json')
    logging.info(f"Loading predefined initial pool from {initial_pool_path}!")
    if os.path.exists(initial_pool_path):
        with open(initial_pool_path, 'r') as f:
            labeled_indices = json.load(f)
        logging.info("Loading successfull!")
    else:
        logging.info(f"ERROR! No predefined initial pool available! Initial pool is randomly sampled. Be aware of random differences between different GPUs!")
        labeled_indices = random.sample(population = [i for i in range(ds_info['n_samples'])], k=args.al.initial_pool_size)
    unlabeled_indices = [i for i in range(ds_info['n_samples']) if i not in labeled_indices]

    # Initialize query
    logging.info(f">>> Initialize Query Strategy {args.al.query_strategy}.")
    query = build_query(args.al.query_strategy)

    # Save Model, Optimizer and LR Scheduler weights for cold start
    initial_model_state = copy.deepcopy(model.state_dict())
    initial_optimizer_state = copy.deepcopy(optimizer.state_dict())
    initial_lrscheduler_state = copy.deepcopy(lr_scheduler.state_dict())

    #Perform Active Learning Cycle
    logging.info(">>> Starting Active Learning Cycles.")
    history = []
    for i_cycle in range(args.al.n_cycles + 1):
        logging.info("[Cycle "+str(i_cycle)+"] Starting Cycle.")

        #If not first cycle, query new data
        if i_cycle != 0:
            logging.info("[Cycle "+str(i_cycle)+"] Query new labels.")
            t1 = time.time()
            newly_labeled_indices = query(
                model=model,
                budget=args.al.query_size,
                labeled_indices=labeled_indices, 
                unlabeled_indices=unlabeled_indices,
                query_dset=query_dataset,
                device=args.device
            )
            labeled_indices += newly_labeled_indices
            unlabeled_indices = [i for i in range(ds_info['n_samples']) if i not in labeled_indices]
            t2 = time.time()
            query_time = t2 - t1
            logging.info(f"[Cycle {i_cycle}] Querying took {round(query_time, 2)} seconds.")

        # Update dataset
        labeled_dset = Subset(dataset, labeled_indices)
        unlabeled_dset = Subset(mixmatch_dataset, unlabeled_indices)

        # Reset Model, Optimizer and Lr Scheduler
        model.load_state_dict(initial_model_state)
        optimizer.load_state_dict(initial_optimizer_state)
        lr_scheduler.load_state_dict(initial_lrscheduler_state)

        # Create DataLoaders
        logging.info("[Cycle "+str(i_cycle)+"] Training Model.")
        labeled_train_loader = DataLoader(labeled_dset, batch_size=args.model.train_batch_size, drop_last=(args.model.train_batch_size < len(labeled_indices)), shuffle=True)
        unlabeled_train_loader = DataLoader(unlabeled_dset, batch_size=args.model.train_batch_size, drop_last=(args.model.train_batch_size < len(labeled_indices)), shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.model.test_batch_size)
        
        # Train model
        train_history = []
        t1 = time.time()
        for i_epoch in range(args.model.n_epochs):
            train_stats = train(labeled_trainloader=labeled_train_loader, unlabeled_trainloader=unlabeled_train_loader, model=model,
                                optimizer=optimizer, criterion=train_criterion, epoch=i_epoch, n_train_iterations=args.ssl.n_train_iterations, device=args.device,
                                T=args.ssl.T, alpha=args.ssl.alpha)
            train_history.append(train_stats)
            lr_scheduler.step()
            if i_epoch % args.al.log_interval == 0 or i_epoch == args.model.n_epochs - 1:
                logging.info(f"[Cycle {i_cycle}][Epoch {i_epoch}] Train Stats: {train_stats}")
        t2 = time.time()
        training_time = t2 - t1
        logging.info(f"[Cycle {i_cycle}] Training took {round(training_time, 2)} seconds.")
        
        # Evaluate Model
        t1 = time.time()
        logging.info("[Cycle "+str(i_cycle)+"] Evaluating Model.")
        test_stats = eval(model, test_loader, eval_criterion)
        t2 = time.time()
        evaluation_time = t2 - t1
        logging.info(f"[Cycle {i_cycle}] Evaluation took {round(evaluation_time, 2)} seconds.")
        logging.info("[Cycle "+str(i_cycle)+"] Test Stats:" + str(test_stats) +".")
        
        #Create Checkpoints
        ckpt = {
            'test_stats' : test_stats,
            'train_history' : train_history,
            'labeled_indices' : labeled_indices,
            'training_time': training_time,
            'querying_time': 0 if i_cycle == 0 else query_time,
            'evaluation_time': evaluation_time
        }
        history.append(ckpt)

    logging.info(">>> Experiment finished, dumping results.")

    # Export results
    with open(os.path.join(args.path.output_dir, 'results.json'), 'w') as f:
        json.dump(history, f)

    # Export final labeled pool
    final_pool_path = os.path.join(args.path.final_pool_dir, 'indices.json')
    logging.info(f"Saving final labeled pool to {final_pool_path}")
    with open(final_pool_path, 'w') as g:
        json.dump(labeled_indices, g)

    logging.info(">>> Finished.")


if __name__ == '__main__':
    main()