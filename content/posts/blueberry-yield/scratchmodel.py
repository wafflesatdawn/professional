import os
from pathlib import Path
import torch, numpy as np, pandas as pd
np.set_printoptions(linewidth=140)
torch.set_printoptions(linewidth=140, sci_mode=False, edgeitems=7)
pd.set_option('display.width', 140)

path = Path('./content/posts/blueberry-yield/data')
df = pd.read_csv(path/'train.csv')

from torch import tensor
t_dep = tensor(df['yield'])
indep_cols = ['id', 'clonesize', 'honeybee', 'bumbles', 'andrena', 'osmia', 'MaxOfUpperTRange', 'MinOfUpperTRange', 'AverageOfUpperTRange',
       'MaxOfLowerTRange', 'MinOfLowerTRange', 'AverageOfLowerTRange', 'RainingDays', 'AverageRainingDays', 'fruitset', 'fruitmass',
       'seeds']
t_indep = tensor(df[indep_cols].values, dtype=torch.float)

# random coefficients to start sgd
n_coeff = t_indep.shape[1]
coeffs = torch.rand(n_coeff) - .5

# test
t_indep*coeffs

# normalization needed. scale all variables using the proportion to the max values
vals, indices = t_indep.max(dim=0)
t_indep = t_indep/vals
t_indep

# generate predictions
preds = (t_indep * coeffs).sum(axis=1)
preds[:4]

# loss function
loss = torch.abs(preds - t_dep).mean()

# wrappers
def calc_preds(coeffs, indeps):
    return (indeps*coeffs).sum(axis=1)
def calc_loss(coeffs, indeps, deps):
    return torch.abs(calc_preds(coeffs, indeps) - deps).mean()

#do a single epoch
coeffs.requires_grad_()
loss = calc_loss(coeffs, t_indep, t_dep)
loss
loss.backward()

r'''doing 2 calls to backward() in a row
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "C:\Users\Welp Windows 10\AppData\Local\Programs\Python\Python311\Lib\site-packages\torch\_tensor.py", line 487, in backward
    torch.autograd.backward(
  File "C:\Users\Welp Windows 10\AppData\Local\Programs\Python\Python311\Lib\site-packages\torch\autograd\__init__.py", line 200, in backward
    Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
RuntimeError: Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed). Saved intermediate values of the graph are freed when you call .backward() or autograd.grad(). Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved tensors after calling backward.'''

with torch.no_grad():
    coeffs.sub_(coeffs.grad * .1) # in pytorch, any method that ends with _ modifies in-place
    coeffs.grad.zero_()
    print(calc_loss(coeffs, t_indep, t_dep))

# before training, split off a validation set
from fastai.data.transforms import RandomSplitter
trn_split, val_split = RandomSplitter(seed=42)(df)
# use indices to subset
trn_indep,val_indep = t_indep[trn_split],t_indep[val_split]
trn_dep,val_dep = t_dep[trn_split],t_dep[val_split]
len(trn_indep),len(val_indep)

# functions for steps taken above
def init_coeffs(n_coeff):
    return (torch.rand(n_coeff) - .5).requires_grad_()

def update_coeffs(coeffs, lr):
    coeffs.sub_(coeffs.grad * lr)
    coeffs.grad.zero_()

def one_epoch(t_dep, t_indep, coeffs, lr=.1):
    loss = calc_loss(coeffs, t_indep, t_dep)
    loss.backward()
    with torch.no_grad():
        update_coeffs(coeffs, lr)
        print(f'{loss: .3f}', end='; ')

def train_model(epochs=30, lr=.1):
    torch.manual_seed(42)
    coeffs = init_coeffs(n_coeff)
    for i in range(epochs):
        one_epoch(t_dep, t_indep, coeffs, lr)
    return coeffs

# test it out
trained = train_model()

# pair columns to values
def show_coeffs(indep_cols, trained):
    return dict(zip(indep_cols, trained.requires_grad_(False)))
show_coeffs(indep_cols, trained)

# defining a metric??
preds = calc_preds(trained, val_indep)

# rewrite using matrix products
def calc_preds(coeffs, indeps):
    return torch.sigmoid(indeps@coeffs)
def init_coeffs(n):
    return (torch.rand(n, 1) * .1).requires_grad_() # second argument to torch.rand(), makes coefficients have one column
# turn dependent variables into column vector by indexing using None, which tells pytorch to add a new dimension
trn_dep = trn_dep[:,None]
val_dep = val_dep[:,None]

# neural network version
def init_coeffs(n_hidden=20):
    layer1 = (torch.rand(n_coeff, n_hidden) - .5) / n_hidden
    layer2 = torch.rand(n_hidden, 1) - .3
    const = torch.rand(1)[0]
    return layer1.requires_grad_(), layer2.requires_grad_(), const.requires_grad_()

import torch.nn.functional as F
def calc_preds(coeffs, indeps):
    l1, l2, const = coeffs
    res = F.relu(indeps@l1)
    res = res@l2 +const
    return torch.sigmoid(res)

def update_coeffs(coeffs, lr):
    for layer in coeffs:
        layer.sub_(layer.grad * lr)
        layer.grad.zero_()

coeffs = train_model(lr=1.4)


# more layers = deep learning
def init_coeffs():
    hiddens = [10,10]
    sizes = [n_coeff] + hiddens + [1]
    n = len(sizes)
    layers = [(torch.rand(sizes[i], sizes[i+1]) - .3) / sizes[i+1]*4 for i in range(n-1)]
    consts = [(torch.rand(1)[0]-.5) * .1 for i in range(n-1)]

def calc_preds(coeffs, indeps):
    layers, consts = coeffs
    n = len(layers)
    res = indeps
    for i,l in enumerate(layers):
        res = res@l + consts[i]
        if i != n-1:
            res = F.relu(res)
    return torch.sigmoid(res)

def update_coeffs(coeffs, lr):
    layers,consts = coeffs
    for layer in layers+consts:
        layer.sub_(layer.grad * lr)
        layer.grad.zero_()

coeffs = train_model(lr=4)