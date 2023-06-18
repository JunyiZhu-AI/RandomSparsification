import torch
import torchvision
import torchvision.transforms as transformers
from opacus.accountants import create_accountant


def dataloader(args, path):

    transform = transformers.Compose([
        transformers.ToTensor(),
        transformers.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root=path, train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=path, train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batchsize//args.batch_partitions, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batchsize//args.batch_partitions, shuffle=True, num_workers=0)

    return trainloader, testloader


def get_epsilon(
        sigma: float,
        sampling_rate: float,
        epochs: int,
        delta: float):
    accountant = create_accountant(mechanism="rdp")
    accountant.steps = [(sigma, sampling_rate, int(epochs / sampling_rate))]
    return accountant.get_epsilon(delta=delta)
