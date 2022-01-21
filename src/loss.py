
import scipy
import numpy as np
from numba import njit

@njit
def logistic_loss_jit(X,y,w):
    pass

@njit
def logistic_loss_grad_jit(X,y,w):
    pass

def logistic_loss(X,y,w):
    t = -y * (X @ w)
    # large enough x approximates log(1+e^x) very well
    # e.g. when x > 23, log(1+e^x) - x < 1e-10
    loss = t*0.0
    loss[t > 20] = t[t > 20]
    loss[t <= 20] = np.log(1 + np.exp(t[t <= 20]))
    return np.mean(loss)

def logistic_loss_grad(X,y,w):
    t = -y * (X @ w)
    r = t*0.0
    ep = np.exp(t[t < 0])
    en = np.exp(-t[t >= 0])
    r[t <  0] = ep / (1 + ep)
    r[t >= 0] = 1 / (1 + en)
    grad = X.T.dot(-y * r) / X.shape[0]
    return grad

def logistic_loss_hessian(X,y,w):
    t = -y * (X @ w)
    r = t*0.0
    ep = np.exp(t[t < 0])
    en = np.exp(-t[t >= 0])
    r[t <  0] = ep / (1 + ep)**2
    r[t >= 0] = en / (1 + en)**2
    hessian = X.T.dot(r.reshape(-1,1) * X) / X.shape[0]
    return hessian

def logistic_loss_hvp(X,y,w,v):
    t = -y * (X @ w)
    r = t*0.0
    ep = np.exp(t[t < 0])
    en = np.exp(-t[t >= 0])
    r[t <  0] = ep / (1 + ep)**2
    r[t >= 0] = en / (1 + en)**2
    # HVP
    # r_n x_nd1 x_nd2 v_d2 = H_d1d2 v_d2 = hvp_d1
    hvp = np.einsum("n,ni,nj,j->i",r,X,X,v) / X.shape[0]
    return hvp


def F(X, y, w, i=None):
    if i is not None:
        if isinstance(i, int):
            i = [i]
        y = y[i]
        X = X[i,:]
    return logistic_loss(X,y,w)

def grad(X, y, w, i=None):
    if i is not None:
        if isinstance(i, int):
            i = [i]
        y = y[i]
        X = X[i,:]
    return logistic_loss_grad(X,y,w)

def hessian(X, y, w, i=None):
    if i is not None:
        if isinstance(i, int):
            i = [i]
        y = y[i]
        X = X[i,:]
    X = np.array(X.todense()) # for element-wise prod
    return logistic_loss_hessian(X,y,w)

def hvp(X, y, w, v, i=None):
    if i is not None:
        if isinstance(i, int):
            i = [i]
        y = y[i]
        X = X[i,:]
    X = np.array(X.todense()) # for element-wise prod
    return logistic_loss_hvp(X,y,w,v)


def test_logistic(X,y,w):
    import torch
    # np
    i = np.random.choice(X.shape[0], 10)
    X = X[i]
    y = y[i]
    L = F(X,y,w)
    G = grad(X,y,w)
    H = hessian(X,y,w)
    v = 10*np.random.rand(X.shape[1])
    Hvp = hvp(X,y,w,v)
    # torch
    def F_torch(w):
    return torch.mean(torch.log(1 + torch.exp(-y * (X @ w))))
    w = torch.Tensor(w).requires_grad_()
    X = torch.Tensor(X.todense()).requires_grad_()
    y = torch.Tensor(y)
    v = torch.Tensor(v)
    L_torch = F_torch(w)
    G_torch, = torch.autograd.grad(L_torch, w)
    H_torch = torch.autograd.functional.hessian(F_torch, w)
    Hvp_torch = torch.autograd.functional.hvp(F_torch, w, v)[1]
    # check
    print(L.sum(), "≈", L_torch.sum().item())
    assert np.abs(L.sum() - L_torch.sum().item()) < 1e-4
    print(G.sum(), "≈", G_torch.sum().item())
    assert np.abs(G.sum() - G_torch.sum().item()) < 1e-4
    print(H.sum(), "≈", H_torch.sum().item())
    assert np.abs(H.sum() - H_torch.sum().item()) < 1e-4
    print(Hvp.sum(), "≈", Hvp_torch.sum().item())
    assert np.abs(Hvp.sum() - Hvp_torch.sum().item()) < 1e-4


if __name__ == "__main__":
    n, d = (500, 50)
    X = np.random.randn(n,d)
    y = np.random.randn(d)
    w = np.random.randn(d)
    test_logistic(X,y,w)
    w = 10*np.random.rand(d)
    test_logistic(X,y,w)
    print("Success!")

