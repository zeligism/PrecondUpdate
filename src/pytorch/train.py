
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import torch.utils.data as data_utils
from joblib import Memory
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import normalize
mem = Memory("./mycache")
DATASET_DIR = "datasets"
LIBSVM_DATASETS = ("a1a", "a9a", "rcv1", "covtype", "real-sim", "w8a", "ijcnn1", "news20",)
OPTIMIZERS = ("SGD", "SARAH", "SVRG", "L-SVRG", "LSVRG", "Adam", "Adagrad", "Adadelta")

@mem.cache
def get_data(filePath):
    data = load_svmlight_file(filePath)
    return data[0], data[1]


def parse_args(namespace=None):
    parser = argparse.ArgumentParser(description="Optimizers with diagonal preconditioning (pytorch)")

    parser.add_argument("-s", "--seed", type=int, default=None,
                        help='Random seed')
    parser.add_argument("--dataset", type=str, choices=DATASETS, default="a1a",
                        help="Name of dataset (in 'datasets' directory)")
    parser.add_argument("-w", "--num_workers", type=int,, default=0,
                        help="Num of data loading workers.")
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
    parser.add_argument("-T", "--epochs", dest="T", type=int, default=5,
                        help="Number of epochs to run.")
    parser.add_argument("-BS", "--batch_size", dest="BS", type=int, default=1,
                        help="Batch size.")
    parser.add_argument("-lr", "--gamma", "--eta", dest="lr", type=float, default=0.02,
                        help="Base learning rate.")
    parser.add_argument("--lr-decay", type=float, default=0,
                        help="Learning rate decay.")
    parser.add_argument("--weight-decay", "--lam", "--lmbd", type=float, default=0,
                        help="weight decay / n")
    parser.add_argument("-p", "--update-p", dest="p", type=float, default=0.999,
                        help="Probability p in L-SVRG or PAGE.")

    parser.add_argument("--precond", type=str.lower, default=None, choices=(None, "hutchinson"),
                        help="Diagonal preconditioner (default: None).")
    parser.add_argument("--beta1", "--momentum", type=float, default=0.9,
                        help="Momentum of gradient first moment.")
    parser.add_argument("--beta2", "--beta", "--rho", dest="beta2", default=0.999,
                        help="Momentum of gradient second moment.")
    parser.add_argument("--alpha", "--eps", type=float, default=1e-7,
                        help="Equivalent to 'eps' in Adam (e.g. see pytorch docs).")
    parser.add_argument("--precond_warmup", type=int, default=100,
                        help="Num of samples for initializing diagonal in hutchinson's method.")
    # parser.add_argument("--precond_resample", action="store_true",
    #                     help="Resample batch in hutchinson's method.")
    # parser.add_argument("--precond_zsamples", type=int, default=1,
    #                     help="Num of rademacher samples in hutchinson's method.")

    # Parse command line args
    args = parser.parse_args(namespace=namespace)

    return args


########## Models ##########
class LogisticRegression(torch.nn.Module):
     def __init__(self, input_dim, output_dim):
         super(LogisticRegression, self).__init__()
         self.linear0 = torch.nn.Linear(input_dim, 100)
         self.linear = torch.nn.Linear(100, output_dim)
     def forward(self, x):
         x = F.relu(self.linear0(x))
         outputs = torch.sigmoid(self.linear(x))
         return outputs

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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
        output = F.log_softmax(x, dim=1)
        return output


########## Datasets ##########
class LibSVMDataset(data_utils.Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        with open(self.dataset, "rb") as f:
            self.X, self.y = load_svmlight_file(f)
            self.X = normalize(self.X, norm='l2', axis=1)
            self.y = (self.y + 1) // 2
        self.num_data, self.feautre_dim = self.X.shape

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        X_tensor = torch.Tensor(self.X[idx].todense()).squeeze(0)
        y_tensor = torch.Tensor([[self.y[idx]]]).squeeze(0)
        return X_tensor, y_tensor


########## Train ##########
def train(model, device, train_loader, optimizer, criterion, epoch, log_interval=50):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        def closure(full_batch=False, create_graph=False):
            optimizer.zero_grad()
            if full_batch:
                for data_inner, target_inner in train_loader:
                    data_inner, target_inner = data_inner.to(device), target_inner.to(device)
                    output = model(data_inner)
                    loss = criterion(output, target_inner) / len(train_loader)
                    loss.backward(create_graph=create_graph)
            else:
                output = model(data)
                loss = criterion(output, target)
                loss.backward(create_graph=create_graph)
            return loss

        loss = optimizer.step(closure)
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader, criterion, multi_class=False):
    model.eval()
    test_loss = 0
    correct = 0
    gradnorm = 0.  # TODO: enable grads to calculate gradnorm?
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss
            if multi_class:
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            else:
                pred = torch.round(output)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    error = 1 - 100. * correct / len(test_loader.dataset)
    return test_loss, error, gradnorm


def init_optim(params, args):
    if args.optimizer == "Adam":
        return optim.Adam(params, lr=args.lr, betas=(args.beta1, args.beta2),
                          eps=args.alpha, weight_decay=args.weight_decay)
    elif args.optimizer == "Adadelta":
        return optim.Adadelta(params, lr=args.lr, rho=args.beta2,
                          eps=args.alpha, weight_decay=args.weight_decay)
    elif args.optimizer == "Adagrad":
        return optim.Adagrad(params, lr=args.lr, lr_decay=args.lr_decay,
                          eps=args.alpha, weight_decay=args.weight_decay)
    elif args.optimizer == "SGD":
        if args.precond == "hutchinson":
            raise NotImplementedError()
        else:
            return optim.SGD(params, lr=args.lr, momentum=args.beta1,
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


def main():
    args = parse_args()

    use_cuda = torch.cuda.is_available() and arg.cuda
    device = torch.device("cuda" if use_cuda else "cpu")

    torch.manual_seed(args.seed)

    if args.dataset == "mnist":
        # Initialize Dataset
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        train_dataset = datasets.MNIST('datasets',
            train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('datasets',
            train=False, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset,
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        test_loader = torch.utils.data.DataLoader(test_dataset,
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        # Initialize model, loss, and optimizer
        model = Net().to(device)
        criterion = F.nll_loss
        args.period = len(train_loader)  # TODO: multiply by a ratio given in args?
        optimizer = init_optim(model.parameters(), args)

        # Train
        for epoch in range(1, args.T + 1):
            train(model, device, train_loader, optimizer, criterion, epoch)
            res = test(model, device, test_loader, criterion, multi_class=True)

    elif args.dataset in LIBSVM_DATASETS:
        # Initialize Dataset
        dataset = LibSVMDataset(args.dataset)
        train_loader = torch.utils.data.DataLoader(dataset,
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        test_loader = torch.utils.data.DataLoader(dataset,
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        
        # Initialize model, loss, and optimizer
        model = LogisticRegression(dataset.feautre_dim, 1).to(device)
        criterion = F.binary_cross_entropy
        args.period = len(train_loader)  # TODO: multiply by a ratio given in args?
        optimizer = init_optim(model.parameters(), args)
        
        # Train
        for epoch in range(1, args.T + 1):
            train(model, device, train_loader, optimizer, criterion, epoch)
            loss, error, gradnorm = test(model, device, test_loader, criterion)


if __name__ == "__main__":
    main()
