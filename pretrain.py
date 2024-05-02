from dataset import build_pretrain_dataset
from model.resnet import ResNet6, ResNet10, ResNet18, ResNet34, ResNet50
from model.simclr import train_one_epoch, evaluate, InfoNCELoss
from utils import seed_everything

import torch
import hydra
import os
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
import torch.optim as optim
import logging
import json
import random
import copy
from omegaconf import OmegaConf


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(args):
    # Print Args for Identification
    logging.info('Using config: \n%s', OmegaConf.to_yaml(args))

    # Create directories necessary for output and model savings
    logging.info(f"Best model will be saved in {args.path.model_dir} !")
    os.makedirs(args.path.model_dir, exist_ok=True)
    os.makedirs(args.path.output_dir, exist_ok=True)

    # Enable Reproducability
    seed_everything(args.random_seed)

    # Data and algorithm specific methods
    logging.info("[Setup] Preparing Dataloaders.")
    dataset, ds_info = build_pretrain_dataset(args)
    val_indices = random.sample(range(len(dataset)), k=args.pretrain.val_split_size)
    train_indices = [i for i in range(len(dataset)) if i not in val_indices]
    trainset = Subset(dataset=dataset, indices=train_indices)
    valset = Subset(dataset=dataset, indices=val_indices)
    trainloader = DataLoader(trainset, batch_size=args.pretrain.batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=args.pretrain.batch_size, shuffle=False)

    # Model (criterion was already build based on algorithm)
    logging.info("[Setup] Building Model.")
    model, optimizer, lr_scheduler, criterion = build_model(args, n_classes=ds_info['n_classes'])

    logging.info("[Setup] Begin Pretraining.")
    history = []
    best_avg_val_loss = None
    for i_epoch in range(args.pretrain.n_epochs):
        # Train Step
        train_stats = train_one_epoch(model=model, dataloader=trainloader, optimizer=optimizer, criterion=criterion, device=args.device)
        lr_scheduler.step()
        logging.info(f"[Epoch {i_epoch}][Training-Results] "+str(train_stats))

        if i_epoch % args.pretrain.val_step_size or i_epoch == args.pretrain.n_epochs -1:
            # Validation Step (Same Loss on a hold out validation split)
            val_stats = evaluate(model=model, dataloader=valloader, criterion=criterion, device=args.device)
            logging.info(f"[Epoch {i_epoch}][Test-Results    ] "+str(val_stats))
            avg_val_loss = val_stats['loss']

            # Save the model with the lowest val loss on its prediction task
            if not best_avg_val_loss or avg_val_loss < best_avg_val_loss:
                logging.info(f"[Epoch {i_epoch}][Model Update    ] Saving new best Model with an average validation loss of {avg_val_loss}.")
                best_avg_val_loss = avg_val_loss
                backbone = reverse_model_change(model=model, n_classes=ds_info['n_classes'])
                state_dict = backbone.state_dict()
                torch.save(state_dict, os.path.join(args.path.model_dir, args.model.name+"_"+str(args.random_seed)+".pth"))
        else:
            val_stats = {}

        # Save Metrics for this epoch
        history.append({
            'val_stats':val_stats,
            'train_stats':train_stats
        })

    logging.info(f">>> Experiment finished. Dumping results.")

    #Export results
    with open(os.path.join(args.path.output_dir, 'results.json'), 'w') as f:
        json.dump(history, f)



def build_model(args, n_classes):
    # Build the base model
    if args.model.name == 'resnet6':
        model = ResNet6(n_classes=n_classes)
    if args.model.name == 'resnet10':
        model = ResNet10(n_classes=n_classes)
    elif args.model.name == 'resnet18':
        model = ResNet18(n_classes=n_classes)
    elif args.model.name == 'resnet34':
        model = ResNet34(n_classes=n_classes)
    elif args.model.name == 'resnet50':
        model = ResNet50(n_classes=n_classes)
    else:
        AssertionError(f"Model {args.model.name} not implemented!")

    # Change Model according to SimCLR algorithm
    input_dim = model.feature_dim
    output_dim = args.pretrain.projection_dim
    model.linear = nn.Identity()
    projector = nn.Sequential(nn.Linear(input_dim, input_dim),
                            nn.ReLU(),
                            nn.Linear(input_dim, output_dim))
    model = nn.Sequential(model, projector)

    optimizer = optim.SGD(model.parameters(), lr=args.pretrain.learning_rate, momentum=0.9, weight_decay=args.pretrain.weight_decay, nesterov=True)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.pretrain.n_epochs)
    criterion = InfoNCELoss()

    return model, optimizer, lr_scheduler, criterion


def reverse_model_change(model, n_classes):
    # Extract the model from the model and undo changes to the last layer
    backbone = copy.deepcopy(model[0])
    backbone.linear = nn.Linear(backbone.feature_dim, n_classes)
    return backbone



if __name__ == '__main__':
    main(None)   