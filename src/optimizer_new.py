
from loss import F, grad, hessian, hessian_diag, hvp, LogisticLoss
from plot import plot_hessian_acc, plot_hessian_approx

import numpy as np


# @TODO: separate into multiple objects
class Preconditioner:
    # Type of possible preconditioners
    TYPES = ("hessian", "hutchinson", "adam", "adagrad", "adadelta", "rmsprop", "momentum")

    def __init__(self, precond_type="hutchinson",
                 beta1=0.0, beta2=0.999, alpha=1e-5,
                 warmup=10, resample=False, zsamples=1,):
        # @TODO: pass loss object?
        self.precond_type = precond_type.lower()
        assert self.precond_type in Preconditioner.TYPES
        self.beta1 = beta1
        self.beta2 = beta2
        self.alpha = alpha
        self.warmup = warmup
        self.resample = resample
        self.zsamples = zsamples
        self.it = None
        self.diagonal = 1.0

    def init(self, loss, w, BS, plot_stats=False):
        self.it = 0
        if self.precond_type == "hessian":
            D = loss.hessian_diag(w)
            D = np.maximum(np.abs(D), self.alpha)
            self.diagonal = D

        elif self.precond_type == "hutchinson":
            assert 0. < self.beta2 and self.beta2 <= 1.
            D_true = loss.hessian_diag(w)
            if plot_stats:
                D_errors = []
            D = 0.
            N = self.warmup * self.zsamples
            for _ in range(self.warmup):
                j = np.random.choice(loss.num_data, BS)
                for _ in range(self.zsamples):
                    z = self.sample_z(w.shape)
                    D += z * loss.hvp(w, z, j) / N
                    if plot_stats:
                        rel_error = np.linalg.norm(D - D_true) / np.linalg.norm(D_true)
                        D_errors.append(rel_error)
            D = np.maximum(np.abs(D), self.alpha)
            self.diagonal = D

        elif self.precond_type == "momentum":
            assert 0. < self.beta1 and self.beta1 <= 1.
            self.m = np.zeros_like(w)

        elif self.precond_type == "rmsprop":
            assert 0. < self.beta2 and self.beta2 <= 1.
            self.v = np.zeros_like(w)

        elif self.precond_type == "adam":
            assert 0. < self.beta1 and self.beta1 <= 1.
            assert 0. < self.beta2 and self.beta2 <= 1.
            self.m = np.zeros_like(w)
            self.v = np.zeros_like(w)

        elif self.precond_type == "adadelta":
            assert 0. < self.beta1 and self.beta1 <= 1.
            assert 0. < self.beta2 and self.beta2 <= 1.
            self.m = np.zeros_like(w)
            self.v = np.zeros_like(w)

        elif self.precond_type == "adagrad":
            self.v = np.zeros_like(w)

        else:
            self.diagonal = 1.0

        # Shows how good the hutchinson approximation is
        if self.precond_type == "hutchinson" and plot_stats:
            plot_hessian_acc(D_true, D)
            plot_hessian_approx(D_errors)
            return

    def update(self, loss, w, i, g):
        self.it += 1
        # Diagonal preconditioning
        if self.precond_type == "hessian":
            D = loss.hessian_diag(w)
            D = np.maximum(np.abs(D), self.alpha)
            self.diagonal = D
            precond_g = self.diagonal**-1 * g

        elif self.precond_type == "hutchinson":
            # estimate hessian diagonal
            D = 0.0
            for _ in range(self.zsamples):
                z = self.sample_z(w.shape)
                # @TODO: try g(w+z) - g(w)?
                D += z * loss.hvp(w, z, i) / self.zsamples
            D = np.abs(self.beta2 * self.diagonal + (1 - self.beta2) * D)
            D = np.maximum(D, self.alpha)
            self.diagonal = D
            precond_g = self.diagonal**-1 * g

        elif self.precond_type == "momentum":
            self.m = self.beta1 * self.m + (1 - self.beta1) * g
            precond_g = self.m

        elif self.precond_type == "rmsprop":
            self.v = self.beta2 * self.v + (1 - self.beta2) * g**2
            self.diagonal = np.sqrt(self.v) + self.alpha
            precond_g = self.diagonal**-1 * g

        elif self.precond_type == "adam":
            self.m = self.beta1 * self.m + (1 - self.beta1) * g
            self.v = self.beta2 * self.v + (1 - self.beta2) * g**2
            m_corr = self.m / (1 - self.beta1 ** self.it)
            v_corr = self.v / (1 - self.beta2 ** self.it)
            self.diagonal = np.sqrt(v_corr) + self.alpha
            precond_g = self.diagonal**-1 * m_corr

        elif self.precond_type == "adadelta":
            self.v = self.beta2 * self.v + (1 - self.beta2) * g**2
            self.diagonal = np.sqrt(self.v) + self.alpha
            precond_g = self.diagonal**-1 * np.sqrt(self.m) * g
            self.m = self.beta1 * self.m + (1 - self.beta1) * precond_g**2

        elif self.precond_type == "adagrad":
            self.v += g**2
            self.diagonal = np.sqrt(self.v) + self.alpha
            precond_g = self.diagonal**-1 * g

        else:
            precond_g = g

        return precond_g

    @staticmethod
    def sample_z(size):
        return 2 * np.random.randint(0, 2, size) - 1


class SGD:
    def __init__(self, loss, w, BS=1, lr=0.0002, history_freq_per_epoch=5):
        self.ep = 0  # effective passes over dataset
        self.w = w
        self.loss = loss
        self.N = self.loss.num_data
        self.BS = BS
        self.lr = lr
        self.history_freq_per_epoch = history_freq_per_epoch
        self.reset_history()
        self.precond = None

    def precondition(self, *args, **kwargs):
        self.precond = Preconditioner(*args, **kwargs)
        return self

    def run(self, T):
        # Initialize preconditioner
        if self.precond is not None:
            self.precond.init(self.loss, self.w, self.BS)
        # Record initial stats
        self.update_history()

        # Run training loop
        num_batches = self.N // self.BS
        effective_iters = T * num_batches
        for it in range(effective_iters):
            # Update step
            self.step()
            # Update history
            freq = self.N // (self.BS * self.history_freq_per_epoch)
            if it % freq == 0:
                self.update_history()

        return self.w, self.history

    def step(self):
        # Grad
        i = np.random.choice(self.N, self.BS)
        self.ep += self.BS / self.N
        g = self.loss.grad(self.w, i)

        # Precond
        if self.precond is not None:
            if self.precond.resample:
                i = np.random.choice(self.N, self.BS)
                self.ep += self.BS / self.N
            g = self.precond.update(self.loss, self.w, i, g)

        self.w -= self.lr * g
        return self

    def reset_history(self):
        self._history = []

    def update_history(self):
        self._history.append(self.stats())

    @property
    def history(self):
        return np.array(self._history)

    def stats(self):
        # loss and gradient
        loss = self.loss.func(self.w)
        g_norm = np.linalg.norm(self.loss.grad(self.w))**2

        # Error
        prediction = self.loss.pred(self.w)
        error = np.mean(prediction < 0)  # wrong prediction -> 100% error
        error += 0.5 * np.mean(prediction == 0)  # ambiguous prediction -> 50% error

        # Preconditioner statistics
        if self.precond is not None:
            D_ratio = np.mean(self.precond.diagonal > self.precond.alpha)
            #H_diag = self.loss.hessian_diag(self.w)
            #H_diag_err = np.linalg.norm(self.precond.diagonal - H_diag) / np.linalg.norm(H_diag)
            H_diag_err = 0.0
        else:
            D_ratio = H_diag_err = 0.0

        return (self.ep, loss, g_norm, error, D_ratio, H_diag_err)


class SVRG(SGD):
    def step(self):
        pass

    def run(self, T):
        return super().run(T)


class SARAH(SGD):
    pass


def SGD_(X, y, T=10000, BS=1, gamma=0.2, beta=0.999, alpha=1e-8, lam=0.0,
         precond="hutchinson", precond_warmup=10, precond_resample=False, precond_zsamples=1,):
    w = np.zeros(X.shape[1])
    loss = LogisticLoss(X,y)
    opt = SGD(loss, w, BS=BS, lr=gamma)
    opt.precondition(precond, beta2=beta, alpha=alpha, warmup=precond_warmup, resample=precond_resample, zsamples=precond_zsamples)
    return opt.run(T)


def Adam_(X, y, T=10000, BS=1, gamma=0.2, beta1=0.9, beta2=0.999, eps=1e-8, **_):
    w = np.zeros(X.shape[1])
    loss = LogisticLoss(X,y)
    opt = SGD(loss, w, BS=BS, lr=gamma).precondition("adam", beta1=beta1, beta2=beta2, alpha=eps)
    return opt.run(T)


