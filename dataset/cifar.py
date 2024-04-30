from torchvision.datasets import CIFAR10
from torchvision import transforms
from model.simclr import RepeatTransformations
from model.mixmatch import TransformTwice

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
    mixmatch_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    colordiststr = 0.5
    color_jitter = transforms.ColorJitter(0.8 * colordiststr, 0.8 * colordiststr, 0.8 * colordiststr, 0.2 * colordiststr)
    contrastive_transform = RepeatTransformations(transforms.Compose([transforms.RandomResizedCrop(size=32), transforms.RandomHorizontalFlip(p=0.5), 
                                                                        transforms.RandomApply([color_jitter], p=0.8), transforms.RandomGrayscale(p=0.2), 
                                                                        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5), transforms.ToTensor(), 
                                                                        transforms.Normalize(mean, std)]))
    eval_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    if split == 'train':
        ds = CIFAR10(path, train=True, download=True, transform=train_transform)
    elif split == 'contrastive_train':
        ds = CIFAR10(path, train=True, download=True, transform=contrastive_transform)
    elif split == 'mixmatch_train':
        ds = CIFAR10(path, train=True, download=True, transform=TransformTwice(mixmatch_transform))
    elif split == 'query' or split == 'test':
        ds = CIFAR10(path, train=True, download=True, transform=eval_transform)
    else:
        AssertionError(f"Split {split} is not defined!")

    if return_info:
        ds_info = {'n_classes': n_classes, 'n_samples':n_samples, 'mean': mean, 'std': std, 'n_channels': n_channels, 'width' : width, 'height': height}
        return ds, ds_info
    return ds