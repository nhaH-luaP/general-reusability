from torchvision.datasets import CIFAR10
from torchvision import transforms

def build_cifar10(split, path='./data', return_info=False):
    mean, std = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.262)
    n_classes = 10
    n_samples = 50000
    n_channels = 3
    width, height = 32, 32

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    if split == 'train':
        ds = CIFAR10(path, train=True, download=True, transform=train_transform)
    elif split == 'query':
        ds = CIFAR10(path, train=True, download=True, transform=eval_transform)
    elif split == 'test':
        ds = CIFAR10(path, train=False, download=True, transform=eval_transform)

    if return_info:
        ds_info = {'n_classes': n_classes, 'n_samples':n_samples, 'mean': mean, 'std': std, 'n_channels': n_channels, 'width' : width, 'height': height}
        return ds, ds_info
    return ds