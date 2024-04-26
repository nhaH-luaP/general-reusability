from torchvision.datasets import MNIST
from torchvision import transforms

def build_mnist(split, path='./data', return_info=False):
    mean, std = (0.1307,), (0.3081,)
    n_classes = 10
    n_samples = 60000
    n_channels = 1
    width, height = 28, 28

    train_transform = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    if split == 'train':
        ds = MNIST(path, train=True, download=True, transform=train_transform)
    elif split == 'query':
        ds = MNIST(path, train=True, download=True, transform=eval_transform)
    elif split == 'test':
        ds = MNIST(path, train=False, download=True, transform=eval_transform)

    if return_info:
        ds_info = {'n_classes': n_classes, 'n_samples':n_samples, 'mean': mean, 'std': std, 'n_channels': n_channels, 'width' : width, 'height': height}
        return ds, ds_info
    return ds