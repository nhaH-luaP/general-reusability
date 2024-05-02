from model.resnet import MiniResNet, ResNet6, ResNet10, ResNet18, ResNet34, ResNet50
from model.wideresnet import WideResnet282, WideResnet2810
from model.utils import train_one_epoch, evaluate
from model.mixmatch import train_one_epoch_mixmatch, SemiLoss

import torch



def build_model(args, ds_info):
    if args.model.name == 'miniresnet':
        model = MiniResNet(n_classes=ds_info['n_classes'])
    elif args.model.name == 'resnet6':
        model = ResNet6(n_classes=ds_info['n_classes'])
    elif args.model.name == 'resnet10':
        model = ResNet10(n_classes=ds_info['n_classes'])
    elif args.model.name == 'resnet18':
        model = ResNet18(n_classes=ds_info['n_classes'])
    elif args.model.name == 'resnet34':
        model = ResNet34(n_classes=ds_info['n_classes'])
    elif args.model.name == 'resnet50':
        model = ResNet50(n_classes=ds_info['n_classes'])
    elif args.model.name == 'wideresnet282':
        model = WideResnet282(n_classes=ds_info['n_classes'])
    elif args.model.name == 'wideresnet2810':
        model = WideResnet2810(n_classes=ds_info['n_classes'])
    else:
        raise NotImplementedError()
    
    if args.ssl.use:
        train = train_one_epoch_mixmatch
        train_criterion = SemiLoss(lambda_u=args.ssl.lambda_u, rampup_length=args.ssl.rampup_length)
    else:
        train = train_one_epoch
        train_criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.model.learning_rate, momentum=args.model.momentum, 
                                nesterov=args.model.nesterov, weight_decay=args.model.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.model.n_epochs)
    eval_criterion = torch.nn.CrossEntropyLoss()
    
    return model, optimizer, lr_scheduler, train, train_criterion, evaluate, eval_criterion