
import numpy as np
import torch
from scipy import sparse

SMALL = 1e-4


def slice(X, y, i):
    if i is None:
        return X, y
    if isinstance(i, int):
        i = [i]
    return X[i,:], y[i]


class LogisticLoss:
    def __init__(self, X, y, weight_decay=0):
        self.X = X
        self.y = y
        self.num_data = X.shape[0]
        self.dim = X.shape[1]
        # @TODO: divide by N?
        self.weight_decay = weight_decay / self.num_data

    def __call__(self, w, i=None):
        self.func(w, i)

    def pred(self, w, i=None):
        # prediction: correct if > 0, wrong otherwise
        X, y = slice(self.X, self.y, i)
        return X.dot(w) * y

    def func(self, w, i=None):
        X, y = slice(self.X, self.y, i)
        loss = logistic_loss(X,y,w)
        if self.weight_decay != 0:
            loss += self.weight_decay * np.linalg.norm(w)**2
        return loss

    def grad(self, w, i=None):
        X, y = slice(self.X, self.y, i)
        g = logistic_loss_grad(X,y,w)
        if self.weight_decay != 0:
            g += self.weight_decay * w
        return g

    def hessian(self, w, i=None):
        X, y = slice(self.X, self.y, i)
        h = logistic_loss_hessian(X,y,w)
        if self.weight_decay != 0:
            h += self.weight_decay * np.identity(like=w)
        return h

    def hessian_diag(self, w, i=None):
        X, y = slice(self.X, self.y, i)
        h_diag = logistic_loss_hessian_diag(X,y,w)
        if self.weight_decay != 0:
            h_diag += self.weight_decay
        return h_diag

    def hvp(self, w, v, i=None):
        X, y = slice(self.X, self.y, i)
        hvp = logistic_loss_hvp(X,y,w,v)
        if self.weight_decay != 0:
            hvp += self.weight_decay * v
        return hvp


class NonLinearLeastSquareLoss:
    def __init__(self, X, y, weight_decay=0):
        self.X = X
        self.y = y
        self.num_data = X.shape[0]
        self.dim = X.shape[1]
        self.weight_decay = weight_decay / self.num_data

    def __call__(w, i=None):
        self.func(w, i)

    def pred(self, w, i=None):
        # prediction: correct if > 0, wrong otherwise
        X, y = slice(self.X, self.y, i)
        return X.dot(w) * y

    def func(self, w, i=None):
        X, y = slice(self.X, self.y, i)
        loss = nlls_loss(X,y,w)
        if self.weight_decay != 0:
            loss += self.weight_decay * np.linalg.norm(w)**2
        return loss

    def grad(self, w, i=None):
        X, y = slice(self.X, self.y, i)
        g = nlls_loss_grad(X,y,w)
        if self.weight_decay != 0:
            g += self.weight_decay * w
        return g

    def hessian(self, w, i=None):
        X, y = slice(self.X, self.y, i)
        h = nlls_loss_hessian(X,y,w)
        if self.weight_decay != 0:
            h += self.weight_decay * np.identity(like=w)
        return h

    def hessian_diag(self, w, i=None):
        X, y = slice(self.X, self.y, i)
        h_diag = nlls_loss_hessian_diag(X,y,w)
        if self.weight_decay != 0:
            h_diag += self.weight_decay
        return h_diag

    def hvp(self, w, v, i=None):
        X, y = slice(self.X, self.y, i)
        hvp = nlls_loss_hvp(X,y,w,v)
        if self.weight_decay != 0:
            hvp += self.weight_decay * v
        return hvp


def logistic_loss(X,y,w, M=25):
    t = -y * (X @ w)
    # x approximates log(1+e^x) well for large x
    # e.g. when x > 23, log(1+e^x) - x < 1e-10
    loss = t*0.0
    loss[t > M] = t[t > M]
    loss[t <= M] = np.log(1 + np.exp(t[t <= M]))
    return np.mean(loss)


def logistic_loss_grad(X,y,w):
    t = -y * (X @ w)
    r = t*0.0
    ep = np.exp(t[t < 0])
    en = np.exp(-t[t >= 0])
    r[t <  0] = ep / (1 + ep)
    r[t >= 0] = 1 / (1 + en)
    grad = X.T @ (-y * r) / X.shape[0]
    return grad


def logistic_loss_hessian(X,y,w):
    t = -y * (X @ w)
    r = t*0.0
    ep = np.exp(t[t < 0])
    en = np.exp(-t[t >= 0])
    r[t <  0] = ep / (1 + ep)**2
    r[t >= 0] = en / (1 + en)**2
    H = X.T @ X.multiply(r.reshape(-1,1)) / X.shape[0]
    return H


def logistic_loss_hessian_diag(X,y,w):
    t = -y * (X @ w)
    r = t*0.0
    ep = np.exp(t[t < 0])
    en = np.exp(-t[t >= 0])
    r[t <  0] = ep / (1 + ep)**2
    r[t >= 0] = en / (1 + en)**2
    H_diag = X.multiply(X.multiply(r.reshape(-1,1))).mean(0)
    return H_diag.A1


def logistic_loss_hvp(X,y,w,v):
    t = -y * (X @ w)
    r = t*0.0
    ep = np.exp(t[t < 0])
    en = np.exp(-t[t >= 0])
    r[t <  0] = ep / (1 + ep)**2
    r[t >= 0] = en / (1 + en)**2
    # HVP
    # r_n x_nd1 x_nd2 v_d2 = H_d1d2 v_d2 = hvp_d1
    #Hvp = np.einsum("n,ni,nj,j->i",r,X,X,v) / X.shape[0]  # requires X.todense()
    Hvp = X.T @ (X.multiply(r.reshape(-1,1)) @ v) / X.shape[0]
    return Hvp


def sigmoid(t):
    ep = np.exp(t[t < 0])
    en = np.exp(-t[t >= 0])
    r = np.zeros_like(t)
    r[t < 0] = ep / (1 + ep)
    r[t >= 0] = 1 / (1 + en)
    return r


def nlls_loss(X, y, w):
    y = (y + 1) / 2  # to {0,1}
    loss = (y - sigmoid(X @ w))**2
    return np.mean(loss)


def nlls_loss_grad(X,y,w):
    y = (y + 1) / 2  # to {0,1}
    s = sigmoid(X @ w)
    r = 2 * (y - s) * s * (1 - s)
    g = -X.T @ r / X.shape[0]
    return g


def nlls_loss_hessian(X,y,w):
    y = (y + 1) / 2  # to {0,1}
    s = sigmoid(X @ w)
    r = (y - 2 * (1 + y) * s + 3 * s**2) * s * (1 - s)
    # h = (y - 2*s - 2*y*s + 3*s**2) * s * (1 - s)
    H = -X.T @ X.multiply(r.reshape(-1,1)) / X.shape[0]
    return H


def nlls_loss_hessian_diag(X,y,w):
    y = (y + 1) / 2  # to {0,1}
    s = sigmoid(X @ w)
    r = (y - 2 * (1 + y) * s + 3 * s**2) * s * (1 - s)
    H_diag = -X.multiply(X.multiply(r.reshape(-1,1))).mean(0)
    return H_diag.A1


def nlls_loss_hvp(X,y,w,v):
    y = (y + 1) / 2  # to {0,1}
    s = sigmoid(X @ w)
    r = (y - 2 * (1 + y) * s + 3 * s**2) * s * (1 - s)
    Hvp = -X.T @ (X.multiply(r.reshape(-1,1)) @ v) / X.shape[0]
    return Hvp


def F(X, y, w, i=None, lam=0.0):
    X, y = slice(X,y,i)
    F = logistic_loss(X,y,w)
    return F + lam * np.linalg.norm(w)**2


def grad(X, y, w, i=None, lam=0.0):
    X, y = slice(X,y,i)
    g = logistic_loss_grad(X,y,w)
    return g + lam * w


def hessian(X, y, w, i=None, lam=0.0):
    X, y = slice(X,y,i)
    H = logistic_loss_hessian(X,y,w)
    return H + lam * np.eye(H.shape[0])


def hessian_diag(X, y, w, i=None, lam=0.0):
    X, y = slice(X,y,i)
    H_diag = logistic_loss_hessian_diag(X,y,w)
    return H_diag + lam


def hvp(X, y, w, v, i=None, lam=0.0):
    X, y = slice(X,y,i)
    Hvp = logistic_loss_hvp(X,y,w,v)
    return Hvp + lam * v


def test_logistic(X,y,w):
    # np
    i = np.random.choice(X.shape[0], 10)
    X = X[i]
    y = y[i]
    L = F(X,y,w)
    G = grad(X,y,w)
    H = hessian(X,y,w)
    H_diag = hessian_diag(X,y,w)
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
    assert np.abs(L.sum() - L_torch.sum().item()) < SMALL
    print(G.sum(), "≈", G_torch.sum().item())
    assert np.abs(G.sum() - G_torch.sum().item()) < SMALL
    print(H.sum(), "≈", H_torch.sum().item())
    assert np.abs(H.sum() - H_torch.sum().item()) < SMALL
    print(Hvp.sum(), "≈", Hvp_torch.sum().item())
    assert np.abs(Hvp.sum() - Hvp_torch.sum().item()) < SMALL
    # H diagonal
    print(H.diagonal().sum(), "≈", H_diag.sum())
    assert np.abs(H.diagonal().sum() - H_diag.sum()) < SMALL


if __name__ == "__main__":
    n, d = (472, 48)
    X = np.random.randn(n,d)
    X = sparse.csr_matrix(X)
    y = 2 * np.random.randint(0,2,[n]) - 1
    test_logistic(X,y,np.random.randn(d))
    test_logistic(X,y,np.random.rand(d))
    print("Success!")

