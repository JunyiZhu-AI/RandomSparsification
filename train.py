import random
import numpy as np
from utils import *
from dpcnn import DPCNN
import argparse
import torch.nn as nn
from opacus.accountants.utils import get_noise_multiplier
from backpack import backpack, extend
from backpack.extensions import BatchGrad

parser = argparse.ArgumentParser()
parser.add_argument('--path-to-data', default="~/data")
parser.add_argument("--epochs", help="Training epoches.", default=40, type=int)
parser.add_argument("--lr", help="Learning rate", default=0.1, type=float)
parser.add_argument("--momentum", help="First order momentum of SGD", default=0, type=float)
parser.add_argument("--batchsize", default=500, help="Data batch-size", type=int)
parser.add_argument("--batch_partitions", default=2, help="Partition data of a batch to reduce memory footprint.", type=int)

# Arguments for DP-SGD with random sparsification
parser.add_argument('--clip', default=1, type=float, help='Clipping bound')
parser.add_argument('--eps', default=3, type=float, help='Privacy variable epsilon')
parser.add_argument('--delta', default=1e-5, type=float, help='Privacy variable delta')
parser.add_argument('--final-rate', default=0, type=float, help='Percentage of parameters get exited finally.')
parser.add_argument('--refresh', default=1, type=int, help='Refresh times of sparsification rate per epoch.')

args = parser.parse_args()
setup = {"device": torch.device("cuda") if torch.cuda.is_available() else "cpu", "dtype": torch.float32}
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)


def main():
    print('==> Loading data..')
    trainloader, testloader = dataloader(args=args, path=args.path_to_data)

    # Computing the privacy budget
    print(f'==> Computing privacy budget..')
    sampling_rate = args.batchsize / len(trainloader.dataset)
    sigma = get_noise_multiplier(target_epsilon=args.eps,
                                 target_delta=args.delta,
                                 sample_rate=sampling_rate,
                                 epochs=args.epochs,)
    print(f'Sigma: {sigma}')

    print(f'==> Building model..')
    net = DPCNN().to(**setup)

    net = extend(net)
    criterion = nn.CrossEntropyLoss(reduction='sum')
    criterion = extend(criterion)

    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)

    num_params = 0
    for p in net.parameters():
        num_params += p.numel()

    best_acc = 0
    for e in range(args.epochs):
        # gradual cooling
        _, _, mask = train_with_rs(trainloader=trainloader,
                                   net=net,
                                   optimizer=optimizer,
                                   criterion=criterion,
                                   sigma=sigma,
                                   clip=args.clip,
                                   epoch=e)

        test_loss, test_acc = test(testloader=testloader,
                                   net=net)

        cur_eps = get_epsilon(sigma=sigma,
                              sampling_rate=sampling_rate,
                              epochs=e+1,
                              delta=args.delta)

        if test_acc > best_acc:
            best_acc = test_acc

        print(f"Epoch: {e+1}/{args.epochs}; Epsilon: {cur_eps:.3f}; RS rate: {len(mask)/num_params:.2f}; "
              f"Test acc.: {test_acc:.1f}; Best acc.: {best_acc:.1f}")


def train_with_rs(trainloader, net, optimizer, criterion, sigma, clip, epoch):
    train_loss_data = 0
    correct = 0
    total = 0

    num_params = 0
    for p in net.parameters():
        num_params += p.numel()

    net.train()
    gradient = torch.zeros(size=[num_params]).to(**setup)
    mini_batch = 0
    for iterations, (data, targets) in enumerate(trainloader):
        # compute current gradual exit rate
        if iterations % (len(trainloader) // args.refresh) == 0:
            rate = np.clip(
                    args.final_rate * (
                        epoch * args.refresh + iterations // (len(trainloader) // args.refresh)
                ) / (args.refresh * args.epochs - 1),0, args.final_rate
            ) if args.epochs >= 0 else 0
            mask = torch.randperm(
                num_params, device=setup['device'], dtype=torch.long
            )[:int(rate * num_params)]

        # training
        optimizer.zero_grad()
        data = data.to(**setup)
        targets = targets.to(device=setup['device'])

        # computing gradients
        pred = net(data)
        loss = criterion(input=pred, target=targets)
        batch_grad = []
        with backpack(BatchGrad()):
            loss.backward()
        for p in net.parameters():
            batch_grad.append(p.grad_batch.reshape(p.grad_batch.shape[0], -1))
            del p.grad_batch

        # clipping gradients
        batch_grad = torch.cat(batch_grad, dim=1)
        for grad in batch_grad:
            grad[mask] = 0
        norm = torch.norm(batch_grad, dim=1)
        scale = torch.clamp(clip/norm, max=1)
        gradient += (batch_grad * scale.view(-1, 1)).sum(dim=0)

        # optimization
        mini_batch += 1
        if mini_batch == args.batch_partitions:
            gradient = gradient / args.batchsize
            mini_batch = 0

            # perturbation
            noise = torch.normal(0, sigma*clip/args.batchsize, size=gradient.shape).to(**setup)
            noise[mask] = 0
            gradient += noise

            # replace non-private gradient with private gradient
            offset = 0
            for p in net.parameters():
                shape = p.grad.shape
                numel = p.grad.numel()
                p.grad.data = gradient[offset:offset + numel].view(shape)
                offset += numel

            optimizer.step()
            gradient = torch.zeros(size=[num_params]).to(**setup)

        train_loss_data += loss.item()/data.shape[0]
        predicted = pred.argmax(1)
        correct += predicted.eq(targets).sum().item()
        total += data.shape[0]

    return train_loss_data/len(trainloader), 100. * correct / total, mask


def test(testloader, net):
    test_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss(reduction='mean')
    net.eval()
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs = inputs.to(**setup)
            targets = targets.to(device=setup['device'])
            pred = net(inputs)
            loss = criterion(pred, targets)
            test_loss += loss.item()
            predicted = pred.argmax(1)
            total += targets.shape[0]
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    return test_loss/len(testloader), acc


if __name__ == '__main__':
    main()

