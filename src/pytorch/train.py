
import time
import argparse
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from joblib import Memory
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import normalize

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import torch.utils.data as data_utils

from scaled_optim import *

mem = Memory("./mycache")
DATASET_DIR = "datasets"
LIBSVM_DATASETS = ("a1a", "a9a", "rcv1", "covtype", "real-sim", "w8a", "ijcnn1", "news20",)
DATASETS = LIBSVM_DATASETS + ("mnist",)
OPTIMIZERS = ("SGD", "SARAH", "SVRG", "L-SVRG", "LSVRG", "Adam", "Adagrad", "Adadelta")
TEST_AT_EPOCH_END = False


def parse_args(namespace=None):
    parser = argparse.ArgumentParser(description="Optimizers with diagonal preconditioning (pytorch)")

    parser.add_argument("-s", "--seed", type=int, default=None,
                        help='Random seed for data')
    parser.add_argument("-m", "--model-seed", type=int, default=0,
                        help='Random seed for model.')
    parser.add_argument("--dataset", type=str, choices=DATASETS, default="a1a",
                        help="Name of dataset (in 'datasets' directory)")
    parser.add_argument("-w", "--num_workers", type=int, default=0,
                        help="Num of data loading workers.")
    parser.add_argument("--cuda", "--gpu", action="store_true", help="Use cuda.")
    # parser.add_argument("--corrupt", nargs="*", type=int, default=None,
    #                     help="Corrupt scale features in dataset."
    #                     "First two args = (k_min, k_max) = range of scaling in powers."
    #                     "If one arg is given, range will be (-k,k).")
    parser.add_argument("--savefig", type=str, default=None,
                        help="Save plots under this name (default: don't save).")
    parser.add_argument("--savedata", type=str, default=None,
                        help="Save data log (default: don't save).")

    parser.add_argument("--optimizer", type=str, choices=OPTIMIZERS, default="SARAH",
                        help="Name of optimizer.")
    parser.add_argument("-T", "--epochs", dest="epochs", type=int, default=5,
                        help="Number of epochs to run.")
    parser.add_argument("-BS", "--batch_size", dest="batch_size", type=int, default=1,
                        help="Batch size.")
    parser.add_argument("-lr", "--gamma", "--eta", dest="lr", type=float, default=0.02,
                        help="Base learning rate.")
    parser.add_argument("--lr-decay", type=float, default=0,
                        help="Learning rate decay.")
    parser.add_argument("--weight-decay", "--lam", "--lmbd", type=float, default=0,
                        help="weight decay / n")
    parser.add_argument("-p", "--ref-prob", dest="p", type=float, default=0.99,
                        help="Probability p in L-SVRG or PAGE.")
    parser.add_argument("--period", type=float, default=1.0,
                        help="Period of checkpoint / inner loop size for SVRG and SARAH (1.0 = one dataset pass)")

    parser.add_argument("--precond", type=str.lower, default=None, choices=(None, "hutchinson"),
                        help="Diagonal preconditioner (default: None).")
    parser.add_argument("--beta1", "--momentum", type=float, default=0.9,
                        help="Momentum of gradient first moment.")
    parser.add_argument("--beta2", "--beta", "--rho", dest="beta2", default=0.999,
                        help="Momentum of gradient second moment.")
    parser.add_argument("--alpha", "--eps", type=float, default=1e-7,
                        help="Equivalent to 'eps' in Adam (e.g. see pytorch docs).")
    parser.add_argument("--warmup", type=int, default=100,
                        help="Num of samples for initializing diagonal in hutchinson's method.")
    # parser.add_argument("--precond_resample", action="store_true",
    #                     help="Resample batch in hutchinson's method.")
    # parser.add_argument("--precond_zsamples", type=int, default=1,
    #                     help="Num of rademacher samples in hutchinson's method.")

    # Parse command line args
    args = parser.parse_args(namespace=namespace)
    if args.beta2 != "avg":
        args.beta2 = float(args.beta2)

    return args


@mem.cache
def get_data(filePath):
    data = load_svmlight_file(filePath)
    return data[0], data[1]


def savedata(data, fname):
    with open(fname, 'wb') as f:
        pickle.dump(data, f)


def savefig(data, fname=None, title="Loss, gradient norm squared, and error"):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.set_size_inches(20, 6)
    plt.suptitle(title)

    ax1.plot(data[:,0], data[:,1])
    ax1.set_ylabel(r"$F(w_t)$")
    ax1.set_xlabel("Effective Passes")
    ax1.grid()

    ax2.semilogy(data[:,0], data[:,2])
    ax2.set_ylabel(r"$||\nabla F(w_t)||^2$")
    ax2.set_xlabel("Effective Passes")
    ax2.grid()

    ax3.semilogy(data[:,0], data[:,3])
    ax3.set_ylabel("Error")
    ax3.set_xlabel("Effective Passes")
    ax3.grid()

    if fname is None:
        plt.show()
    else:
        plt.savefig(fname)
        plt.close()


########## Models ##########
class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear0 = torch.nn.Linear(input_dim, 100)
        self.linear = torch.nn.Linear(100, output_dim)

    def forward(self, x):
        x = F.relu(self.linear0(x))
        outputs = torch.sigmoid(self.linear(x))
        return outputs


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


# https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)  # add padding
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


########## Datasets ##########
class LibSVMDataset(data_utils.Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        with open(self.dataset, "rb") as f:
            self.X, self.y = load_svmlight_file(f)
            self.X = normalize(self.X, norm='l2', axis=1)
            self.y = (self.y + 1) // 2
        self.num_data, self.feature_dim = self.X.shape

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        X_tensor = torch.Tensor(self.X[idx].todense()).squeeze(0)
        y_tensor = torch.Tensor([[self.y[idx]]]).squeeze(0)
        return X_tensor, y_tensor


########## Train ##########
def test(model, device, test_loader, criterion, multi_class=False, show_results=False):
    model.eval()
    test_loss = 0
    correct = 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        loss = criterion(y_pred, y) / len(test_loader)
        loss.backward()
        test_loss += loss.item()
        # Accuracy
        if multi_class:
            pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        else:
            pred = torch.round(y_pred)
        correct += pred.eq(y.view_as(pred)).sum().item()

    gradnorm = 0.0
    for i, p in enumerate(model.parameters()):
        if p.grad is not None:
            gradnorm += torch.sum(p.grad**2).item()
            p.grad.detach_()
            p.grad.zero_()
    gradnorm = gradnorm**0.5

    if show_results:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    error = (len(test_loader.dataset) - correct) / len(test_loader.dataset)
    return test_loss, gradnorm, error


def train(model, device, train_loader, test_loader, optimizer, criterion, epoch,
          log_interval=50, test_interval=0.25, multi_class=False):
    model.train()
    data = []
    # Add test at initial point
    if epoch == 1:
        result0 = test(model, device, test_loader, criterion, multi_class=multi_class)
        data.append((0.,) + result0)

    # Training loop
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)

        def closure(full_batch=False, create_graph=False):
            optimizer.zero_grad()
            if full_batch:
                for x_inner, y_inner in train_loader:
                    x_inner, y_inner = x_inner.to(device), y_inner.to(device)
                    y_pred_inner = model(x_inner)
                    loss = criterion(y_pred_inner, y_inner) / len(train_loader)
                    loss.backward(create_graph=create_graph)
            else:
                y_pred = model(x)
                loss = criterion(y_pred, y)
                loss.backward(create_graph=create_graph)
            return loss

        loss = optimizer.step(closure)
        # Logging
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(x), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        # Testing
        should_test = (batch_idx + 1) % round(test_interval * len(train_loader)) == 0
        last_epoch = batch_idx == len(train_loader) - 1
        if should_test or last_epoch:
            ep = epoch - 1 + (batch_idx + 1) / len(train_loader)
            # XXX: Ugly hack but whatever
            if hasattr(optimizer, 'global_state') and 'ref_evals' in optimizer.global_state:
                ep += optimizer.global_state['ref_evals']
            # Show results if last batch
            result = test(model, device, test_loader, criterion,
                          multi_class=multi_class, show_results=last_epoch)
            data.append((ep,) + result)

    return data


def init_optim(params, args):
    if args.optimizer == "Adam":
        optimizer = optim.Adam(params, lr=args.lr, betas=(args.beta1, args.beta2),
                               eps=args.alpha, weight_decay=args.weight_decay)
    elif args.optimizer == "Adadelta":
        optimizer = optim.Adadelta(params, lr=args.lr, rho=args.beta2,
                                   eps=args.alpha, weight_decay=args.weight_decay)
    elif args.optimizer == "Adagrad":
        optimizer = optim.Adagrad(params, lr=args.lr, lr_decay=args.lr_decay,
                                  eps=args.alpha, weight_decay=args.weight_decay)
    elif args.optimizer == "SGD":
        if args.precond == "hutchinson":
            optimizer = ScaledSGD(params, lr=args.lr, beta=args.beta2, alpha=args.alpha,
                                  warmup=args.warmup)
        else:
            optimizer = optim.SGD(params, lr=args.lr, momentum=args.beta1,
                                  weight_decay=args.weight_decay)
    elif args.optimizer == "SVRG":
        if args.precond == "hutchinson":
            optimizer = ScaledSVRG(params, lr=args.lr, period=args.period,
                                   beta=args.beta2, alpha=args.alpha, warmup=args.warmup)
        else:
            raise NotImplementedError()
    elif args.optimizer == "SARAH":
        if args.precond == "hutchinson":
            optimizer = ScaledSARAH(params, lr=args.lr, period=args.period,
                                    beta=args.beta2, alpha=args.alpha, warmup=args.warmup)
        else:
            raise NotImplementedError()
    elif args.optimizer in ("LSVRG", "L-SVRG"):
        if args.precond == "hutchinson":
            optimizer = ScaledLSVRG(params, lr=args.lr, p=args.p,
                                    beta=args.beta2, alpha=args.alpha, warmup=args.warmup)
        else:
            raise NotImplementedError()
    return optimizer


def run(args):
    use_cuda = torch.cuda.is_available() and args.cuda
    use_mps = torch.backends.mps.is_available() and args.cuda
    device = torch.device("cuda" if use_cuda else ("mps" if use_mps else "cpu"))
    if use_cuda:
        print(f"Using CUDA")
    if use_mps:
        print(f"Using M1")

    print(f"Dataset: {args.dataset}")
    print(f"Num workers: {args.num_workers}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {device}")

    #---------- MNIST ----------#
    if args.dataset == "mnist":
        # Initialize Dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(DATASET_DIR, train=True,
                                       download=True, transform=transform)
        test_dataset = datasets.MNIST(DATASET_DIR, train=False,
                                      transform=transform)

        # All runs start are init based on the model seed
        print(f"Initializing model with random seed {args.model_seed}.")
        torch.manual_seed(args.model_seed)
        # model = Net().to(device)
        model = LeNet5().to(device)

        # Now set the random seed
        if args.seed is not None:
            print(f"Setting random seed to {args.seed}.")
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)

        # Initialize DataLoaders
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                   shuffle=True, num_workers=args.num_workers)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                  shuffle=False, num_workers=args.num_workers)

        args.period = round(args.period * len(train_loader))  # change period to num batches

        # Loss and optimizer
        criterion = F.cross_entropy
        optimizer = init_optim(model.parameters(), args)

        # Train
        data = []
        for epoch in range(1, args.epochs + 1):
            results = train(model, device, train_loader, test_loader,
                            optimizer, criterion, epoch, multi_class=True)
            data += results
            # Test at the end to show results
            if TEST_AT_EPOCH_END:
                test(model, device, test_loader, criterion, multi_class=True, show_results=True)

    #---------- LIBSVM ----------#
    elif args.dataset in LIBSVM_DATASETS:
        # Initialize Dataset
        dataset_path = os.path.join(DATASET_DIR, args.dataset)
        if not os.path.isfile(dataset_path):
            raise FileNotFoundError(f"Could not find dataset at '{dataset_path}'")
        libsvm_dataset = LibSVMDataset(dataset_path)

        # All runs start are init based on the model seed
        print(f"Initializing model with random seed {args.model_seed}.")
        torch.manual_seed(args.model_seed)
        model = LogisticRegression(libsvm_dataset.feature_dim, 1).to(device)

        # Now set random seed
        if args.seed is not None:
            print(f"Setting random seed to {args.seed}.")
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)

        # Initialize DataLoaders
        train_loader = torch.utils.data.DataLoader(libsvm_dataset, batch_size=args.batch_size,
                                                   shuffle=True, num_workers=args.num_workers)
        test_loader = torch.utils.data.DataLoader(libsvm_dataset, batch_size=args.batch_size,
                                                  shuffle=False, num_workers=args.num_workers)

        args.period = round(args.period * len(train_loader))  # change period to num batches

        # Loss and optimizer
        criterion = F.binary_cross_entropy
        optimizer = init_optim(model.parameters(), args)

        # Train
        data = []
        for epoch in range(1, args.epochs + 1):
            results = train(model, device, train_loader, test_loader, optimizer, criterion, epoch)
            data += results
            # Test at the end to show results
            if TEST_AT_EPOCH_END:
                test(model, device, test_loader, criterion, show_results=True)

    # Show results
    if args.savefig is not None:
        savefig(np.array(data), args.savefig)

    if args.savedata is not None:
        print(f"Saving data to '{args.savedata}'.")
        savedata(data, args.savedata)


def main():
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
