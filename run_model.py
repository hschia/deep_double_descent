import numpy as np
import matplotlib.pyplot as plt
import math
import pickle
import json
import yaml
import sys
import os

import torch
from torch import nn
from torch.utils import data

import torchvision
from torchvision import transforms

from IPython import display
from d2l import torch as d2l

print(f'GPU available: {torch.cuda.is_available()}')


def training_set_labelnoise(train_data, noise_ratio=0.2):
    
    # create deepcopy
    import copy
    noisy = copy.deepcopy(train_data)
    noisy.data = copy.deepcopy(train_data.data)
    noisy.targets = copy.deepcopy(train_data.targets)
    
    n_classes = len(noisy.classes)
    targets = noisy.targets
    targets_set = targets.unique()
    
    # generate mislabels for each cls
    for cls in range(n_classes):
        
        cls_mask = (targets == cls)
        targets_filtered = np.array(targets[cls_mask])
        
        wrong_target_cls = [x for x in targets_set if x != cls] # labels other than cls
        n_mislabels = int(len(targets_filtered)*noise_ratio) # number of mislabels for cls
        
        wrong_targets = np.random.choice(wrong_target_cls, size=n_mislabels)
        random_boolean = np.random.permutation([True]*n_mislabels + [False]*(len(targets_filtered) - n_mislabels))
        targets_filtered[random_boolean] = wrong_targets
        targets[cls_mask] = torch.tensor(targets_filtered)
        
    noisy.targets = targets
    
    return noisy

def load_data_fashion_mnist(batch_size, resize=None, num_workers=8):
    
    # Use [] iterable because we use transform.Compose and possibly insert below
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize)) # Add resizing as optional transformation
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root='../data', train=True, transform=trans, download=True, target_transform=None)
    mnist_test = torchvision.datasets.FashionMNIST(
        root='../data', train=False, transform=trans, download=True, target_transform=None)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=num_workers),
            data.DataLoader(mnist_test, batch_size, shuffle=True, num_workers=num_workers)
           )

def load_data_mnist(batch_size, train_size=None, test_size=None, label_noise=False, noise_ratio=0.2, resize=None, num_workers=8):
    
    # Use [] iterable because we use transform.Compose and possibly insert below
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize)) # Add resizing as optional transformation
    trans = transforms.Compose(trans)
    
    mnist_train = torchvision.datasets.MNIST(
        root='../data', train=True, transform=trans, download=True, target_transform=None)
    mnist_test = torchvision.datasets.MNIST(
        root='../data', train=False, transform=trans, download=True, target_transform=None)
    
    if label_noise:
        mnist_train = training_set_labelnoise(mnist_train, noise_ratio=noise_ratio)
    
    if train_size is not None:
        indices = np.random.choice(len(mnist_train), size=train_size, replace=False)
        mnist_train = data.Subset(mnist_train, indices)
        
    if test_size is not None:
        indices = np.random.choice(len(mnist_test), size=test_size, replace=False)
        mnist_test = data.Subset(mnist_test, indices)
    
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=num_workers),
            data.DataLoader(mnist_test, batch_size, shuffle=True, num_workers=num_workers)
           )

def try_gpu():
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    else:
        return torch.device('cpu')
    

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend, title):
    """Set the axes for matplotlib."""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    axes.set_title(title)
    if legend:
        axes.legend(legend)
    axes.grid()

class Animator:
    """For plotting data in animation."""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, title=None, xscale='linear', yscale='log',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # Incrementally plot multiple lines
        if legend is None:
            legend = []
        display.set_matplotlib_formats('svg')
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes,]
        # Use a lambda function to capture arguments
        self.config_axes = lambda: set_axes(self.axes[
            0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend, title)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # Add multiple data points into the figure
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)

        
class Accumulator:
    def __init__(self, n):
        self.data = [0.0]*n
        
    def add(self, *args):
        self.data = [a + b for a, b in zip(self.data, args)]
        
    def __getitem__(self, idx):
        return self.data[idx]

    
def train(net, train_iter, test_iter, num_epochs, loss, optimizer, device):
    
    # initialize weights
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.kaiming_normal_(m.weight)
    net.apply(init_weights)
    
    # setup
    # optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    # loss = nn.CrossEntropyLoss()
    num_batches = len(train_iter)
    net.to(device)
    
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'test loss'], title=f'hidden layer: {net[1]}')
    
    train_loss_ls, test_loss_ls = [], []
    for epoch in range(num_epochs):
        
        if epoch % 100 == 0:
            print('Current epoch:', epoch)
        
        # train loop
        net.train()
        metric_train = Accumulator(2)
        for batch, (X, y) in enumerate(train_iter, 1):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            yhat = net(X)
            l = loss(yhat, y)
            l.backward()
            optimizer.step()
            
            with torch.no_grad():
                metric_train.add(l*y.numel(), y.numel())
            train_loss = metric_train[0]/metric_train[1]
            train_loss_ls.append(train_loss)
                
        # test loop
        net.eval()
        metric_test = Accumulator(2)
        for X, y in test_iter:
            with torch.no_grad():
                X, y = X.to(device), y.to(device)
                yhat = net(X)
                l = loss(yhat, y)
                metric_test.add(l*y.numel(), y.numel())
                
                test_loss = metric_test[0]/metric_test[1]
                test_loss_ls.append(test_loss)
                
        animator.add(epoch+1, (train_loss, test_loss))
    
    return animator, train_loss_ls, test_loss_ls



######## Settings ########

num_epochs = 500
train_size = 2000
test_size = 10000
batch_size = 256

# Create label noise or not
label_noise = True
noise_ratio = 0.1

## Create file and directory paths
if label_noise:
    DIRPATH = f'./double_descent/mnist_train{train_size}_test{test_size}_epoch{num_epochs}_labelnoise{int(noise_ratio*100)}'
else:
    DIRPATH = f'./double_descent/mnist_train{train_size}_test{test_size}_epoch{num_epochs}'

if not os.path.isdir(DIRPATH):
    os.makedirs(DIRPATH)
    os.system(f"chmod 777 {DIRPATH}")

lr = 0.1
loss = nn.CrossEntropyLoss()
num_workers = 1

train_iter, test_iter = load_data_mnist(batch_size, train_size=train_size, test_size=test_size, label_noise=label_noise, noise_ratio=noise_ratio, num_workers=num_workers)


# Key in input in Jupyter notebook
d_hidden = int(sys.argv[1])

net = nn.Sequential(nn.Flatten(), # needed to convert 2D images to 1D array
                    nn.Linear(28*28, d_hidden), 
                    nn.ReLU(),
                    nn.Linear(d_hidden,10))

print('Running dhidden: ', d_hidden)

optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0)    
animator, train_loss_ls, test_loss_ls = train(net, train_iter, test_iter, num_epochs, loss, optimizer, try_gpu())

PATH = os.path.join(DIRPATH, f'd_hidden{d_hidden}')
MODELPATH = os.path.join(DIRPATH, f'd_hidden{d_hidden}.pt')
FIGPATH = PATH + '.png'
PKLPATH = os.path.join(DIRPATH, 'data.pkl')

torch.save(net, MODELPATH)
animator.fig.savefig(FIGPATH, bbox_inches='tight')

# If file does not exist, create empty dic
if not os.path.isfile(PKLPATH):
    dhidden_train_test = {}
    
    with open(PKLPATH, 'wb') as f:
        pickle.dump(dhidden_train_test, f)

# Load, rewrite and save file
with open(PKLPATH, 'rb') as f:
    dhidden_train_test = pickle.load(f)

dhidden_train_test[d_hidden] = (train_loss_ls[-1], test_loss_ls[-1])

with open(PKLPATH, 'wb') as f:
    pickle.dump(dhidden_train_test, f)