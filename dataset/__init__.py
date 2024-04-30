from dataset.cifar import build_cifar10



def build_dataset(args):
    if args.dataset == 'cifar10':
        dataset, ds_info = build_cifar10(split='train', path=args.path.data_dir, return_info=True)
        mixmatch_dataset = build_cifar10(split='mixmatch_train', path=args.path.data_dir)
        query_dataset = build_cifar10(split='query', path=args.path.data_dir)
        test_dataset = build_cifar10(split='test', path=args.path.data_dir)
    else:
        raise NotImplementedError()
    return dataset, mixmatch_dataset, query_dataset, test_dataset, ds_info

def build_pretrain_dataset(args):
    if args.dataset == 'cifar10':
        dataset, ds_info = build_cifar10(split='contrastive_train', path=args.path.data_dir, return_info=True)
    return dataset, ds_info