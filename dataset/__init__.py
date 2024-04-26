from dataset.cifar import build_cifar10
from dataset.mnist import build_mnist

def build_dataset(args):
    if args.dataset == 'cifar10':
        dataset, ds_info = build_cifar10(split='train', path=args.path.data_dir, return_info=True)
        query_dataset = build_cifar10(split='query', path=args.path.data_dir)
        test_dataset = build_cifar10(split='test', path=args.path.data_dir)
    elif args.dataset == 'mnist':
        dataset, ds_info = build_mnist(split='train', path=args.path.data_dir, return_info=True)
        query_dataset = build_mnist(split='query', path=args.path.data_dir)
        test_dataset = build_mnist(split='test', path=args.path.data_dir)
    else:
        raise NotImplementedError()
    return dataset, query_dataset, test_dataset, ds_info