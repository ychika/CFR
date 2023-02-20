import sys
import numpy as np
import torch
from util import *

def objective(y, y_hat, X, a, p, reg_alpha, loss_func='l1', imbalance_func='mmd2_lin', bool_weighting=True):
    # compute sample weights
    w = sample_weights(a, p) if bool_weighting else 1.0

    # compute prediction loss
    if loss_func == 'l1':
        pred_loss = torch_l1_loss(y, y_hat, w)
    elif loss_func == 'ce':
        pred_loss = torch_ce_loss(y, y_hat, w)
    else:
        print("Error: loss_func = " + str(loss_func) + " is not implemented", file=sys.stderr)
        sys.exit(1)

    # compute 'imbalance loss' used to balance feature representations
    if imbalance_func == 'lin_disc':
        imbalance_loss = lin_disc(X, a, p)
    elif imbalance_func == 'mmd2_lin':
        imbalance_loss = mmd2_lin(X, a, p)
    elif imbalance_func == 'mmd2_rbf':
        sig = 0.1 # Band-width value for IHDP dataset (according to original cfrnet implementation)
        imbalance_loss = mmd2_rbf(X, a, p, sig)
    elif imbalance_func == 'wasserstein':
        imbalance_loss = wasserstein(X, a, p)
    else:
        print("Error: imbalance_func = " + str(imbalance_func) + " is not implemented", file=sys.stderr)
        sys.exit(1)    

    return pred_loss + reg_alpha * imbalance_loss

##### Loss functions
### (weighted) L1 loss for continuous-valued outcome
def torch_l1_loss(y, y_hat, w):
    res = w * torch.abs(y_hat - y)
    return torch.mean(res)
### (weighted) cross entropy loss for binary outcome
def torch_ce_loss(y, y_hat, w):
    y_hat = 0.995 / (1.0 + torch.exp(- y_hat)) + 0.0025 # ?
    res = w * (y * torch.log(y_hat) + (1.0 - y) * torch.log(1.0 - y_hat))
    return torch.mean(res)
### sample weighting [Shalit+; Eq. (3), Sec.4, ICML2017]
def sample_weights(a, p):
    w1 = a / (2 * p)
    w0 = (1 - a) / (2 * (1 - p))
    return w0 + w1 

##### Weight decay regularization [Johansson+; Sec.6, ICML2016], [Shalit+; Sec.5, ICML2017]

##### Balancing penalty functions
### Linear Discrepancy [Johansson+; Eq. (8), Sec. 4.1, ICML2016]
def lin_disc(X, a, p):
    mmd = mmd2_lin(X, a, p, const=1.0)
    sign_p = np.sign(p - 0.5)
    disc = sign_p * (p - 0.5) + torch_safe_sqrt(np.square(2 * p - 1) * 0.25 + mmd)
    return disc

### Linear MMD [Johansson+; ||v|| const=2.0 leads to ||v|| in Eq. (8), Sec. 4.1, ICML2016]

def mmd2_lin(X, a, p, const=2.0):
    #_t_ind = torch.where((a == 1).all(axis=1))[0] ## a: treatment vector in torch.Tensor
    #_c_ind = torch.where((a == 0).all(axis=1))[0]
    _t_ind = torch.where(a == 1)[0] ## a: treatment vector in torch.Tensor
    _c_ind = torch.where(a == 0)[0]
    X1 = X[_t_ind]
    X0 = X[_c_ind]
    X1_mean = torch.mean(X1, dim=0)
    X0_mean = torch.mean(X0, dim=0)
    mmd = torch.sum(torch.square(const * p * X1_mean - const * (1.0 - p) * X0_mean))
    return mmd

### MMD with Gaussian kernel [Shalit+; Appendix B.2, ICML2017]
def mmd2_rbf(X, a, p, sig):
    _t_ind = torch.where(a == 1)[0] ## a: treatment vector in torch.Tensor
    _c_ind = torch.where(a == 0)[0]
    X1 = X[_t_ind]
    X0 = X[_c_ind]
    num_samples_0 = float(X0.shape[0])
    num_samples_1 = float(X1.shape[0])
    # compute gram matrices
    Gx_00 = torch.exp(-torch_pdist2(X0, X0) / (sig ** 2))
    Gx_01 = torch.exp(-torch_pdist2(X0, X1) / (sig ** 2))
    Gx_11 = torch.exp(-torch_pdist2(X1, X1) / (sig ** 2))
    mmd = np.square(1.0 - p) / (num_samples_0 * (num_samples_0 - 1.0)) * (torch.sum(Gx_00) - num_samples_0)
    + np.square(p) / (num_samples_1 * (num_samples_1 - 1.0)) * (torch.sum(Gx_11) - num_samples_1)
    - 2.0 * p * (1.0 - p) / (num_samples_0 * num_samples_1) * torch.sum(Gx_01)
    mmd = 4.0 * mmd
    return mmd   

### Approximated Wasserstein distance with Sinkhorn [Shalit+; Appendix B.1, ICML2017]
def wasserstein(X, a, p, lamb=10, its=10, sq=False, backpropT=False):
    _t_ind = torch.where(a == 1)[0] ## a: treatment vector in torch.Tensor
    _c_ind = torch.where(a == 0)[0]
    X1 = X[_t_ind]
    X0 = X[_c_ind]
    num_samples_0 = float(X0.shape[0])
    num_samples_1 = float(X1.shape[0])
    # compute distance matrix
    Mx_10 = torch_pdist2(X1, X0)
    if sq is False:
        Mx_10 = torch_safe_sqrt(Mx_10)
    # estimate lambda & delta
    Mx_10_mean = torch.mean(Mx_10)
    #torch_dropout = torch.nn.dropout(10 / (num_samples_0 * num_samples_1))
    #Mx_10_drop = torch_dropout(Mx_10)
    delta = torch.max(Mx_10).detach() # detach() = no gradient computed
    eff_lamb = (lamb / Mx_10_mean).detach()
    # compute new distance matrix
    Mx_10_new = Mx_10
    row = delta * torch.ones(Mx_10[0:1,:].shape)
    col = torch.cat([delta * torch.ones(Mx_10[:,0:1].shape), torch.zeros((1,1))], 0)
    Mx_10_new = torch.cat([Mx_10, row], 0)
    Mx_10_new = torch.cat([Mx_10_new, col], 1)
    # compute marginal vectors
    marginal1 = torch.cat([p * torch.ones(torch.where(a == 1)[0].reshape((-1, 1)).shape) / num_samples_1, (1-p) * torch.ones((1,1))], 0)
    marginal0 = torch.cat([(1 - p) * torch.ones(torch.where(a == 0)[0].reshape((-1, 1)).shape) / num_samples_0, p * torch.ones((1,1))], 0)
    # compute kernel matrix
    Mx_10_lamb = eff_lamb * Mx_10_new
    Kx_10 = torch.exp(- Mx_10_lamb) + 1.0e-06 ## constant added to avoid nan
    U = Kx_10 * Mx_10_new
    marginal1invK = Kx_10 / marginal1
    # fixed-point iterations of Sinkhorn algorithm [Cuturi+; NeurIPS2013]
    u = marginal1
    for i in range(0, its):
        u = 1.0 / (torch.matmul(marginal1invK, (marginal0 / torch.t(torch.matmul(torch.t(u), Kx_10)))))
    v = marginal0 / (torch.t(torch.matmul(torch.t(u), Kx_10)))

    T = u * (torch.t(v) * Kx_10)

    if backpropT is False:
        T = T.detach()

    E = T * Mx_10_new
    D = 2 * torch.sum(E)
    return D#, Mx_10_lamb
