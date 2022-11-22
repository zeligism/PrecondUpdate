
import numpy as np
from loss import LogisticLoss
from plot import plot_hessian_acc, plot_hessian_approx


def sample_uniform(size=1):
    return np.random.rand(size)


def sample_bernoulli(size=1):
    return np.random.randint(0, 2, size)


# @TODO: separate into multiple objects
class Preconditioner:
    # Type of possible preconditioners
    TYPES = ("none", "hessian", "hutchinson", "hutch++", "adam", "adagrad", "adadelta", "rmsprop", "momentum")

    def __init__(self, precond="hutchinson",
                 beta1=0.0, beta2=0.999, alpha=1e-5,
                 precond_warmup=10, precond_resample=False, precond_zsamples=1,):
        self.precond_type = "none" if precond is None else precond.lower()
        assert self.precond_type in Preconditioner.TYPES
        self.beta1 = beta1
        self.beta2 = beta2
        self.alpha = alpha
        self.warmup = precond_warmup
        self.resample = precond_resample
        self.zsamples = precond_zsamples
        self.t = 0
        self.diagonal = 1.0

    def init(self, w, loss, BS, plot_stats=False):
        # @TODO: option for plot_stats
        self.t = 0
        if self.precond_type == "hessian":
            D = loss.hessian_diag(w)
            D = np.maximum(np.abs(D), self.alpha)
            self.diagonal = D

        elif self.precond_type == "hutchinson":
            assert self.beta2 == "avg" or (0. <= self.beta2 and self.beta2 <= 1.)
            if plot_stats:
                H_diag = loss.hessian_diag(w)
                D_errors = []
            D = 0.
            N = self.warmup * self.zsamples
            for _ in range(self.warmup):
                i = np.random.choice(loss.num_data, BS)
                for _ in range(self.zsamples):
                    z = 2 * sample_bernoulli(w.shape) - 1
                    D += z * loss.hvp(w, z, i) / N
                    if plot_stats:
                        rel_error = np.linalg.norm(D - H_diag) / np.linalg.norm(H_diag)
                        D_errors.append(rel_error)
            D = np.maximum(np.abs(D), self.alpha)
            self.diagonal = D

        elif self.precond_type == "hutch++":
            assert self.beta2 == "avg" or (0. <= self.beta2 and self.beta2 <= 1.)
            D = 0.
            for _ in range(self.warmup):
                m = self.zsamples
                i = np.random.choice(loss.num_data, BS)
                S = np.random.randn(w.shape[0], m)
                G = np.random.randn(w.shape[0], m)
                hvp_S = loss.hvp(w, S, i)
                Q, _ = np.linalg.qr(hvp_S)
                hvp_Q = loss.hvp(w, Q, i)
                P = G - Q.dot(Q.T.dot(G))
                hvp_P = loss.hvp(w, P, i)
                H1 = Q * hvp_Q
                H2 = P * hvp_P
                D += (H1.sum(1) + H2.sum(1) / m) / self.warmup
            D = np.maximum(np.abs(D), self.alpha)
            self.diagonal = D

        elif self.precond_type == "momentum":
            assert 0. <= self.beta1 and self.beta1 <= 1.
            self.m = np.zeros_like(w)

        elif self.precond_type == "rmsprop":
            assert 0. <= self.beta2 and self.beta2 <= 1.
            self.v = np.zeros_like(w)

        elif self.precond_type == "adam":
            assert 0. <= self.beta1 and self.beta1 <= 1.
            assert 0. <= self.beta2 and self.beta2 <= 1.
            self.m = np.zeros_like(w)
            self.v = np.zeros_like(w)

        elif self.precond_type == "adadelta":
            assert 0. <= self.beta2 and self.beta2 <= 1.
            self.v = np.zeros_like(w)
            self.u = np.zeros_like(w)

        elif self.precond_type == "adagrad":
            self.v = np.zeros_like(w)

        else:
            self.diagonal = np.ones_like(w)

        # Shows how good the hutchinson approximation is
        if self.precond_type == "hutchinson" and plot_stats:
            plot_hessian_acc(H_diag, D)
            plot_hessian_approx(D_errors)

    def update(self, w, loss, i, g):
        self.t += 1
        # Diagonal preconditioning
        if self.precond_type == "hessian":
            D = loss.hessian_diag(w)
            D = np.maximum(np.abs(D), self.alpha)
            self.diagonal = D
            precond_g = self.diagonal**-1 * g

        elif self.precond_type == "hutchinson":
            # estimate hessian diagonal
            D = 0.0
            averaging_beta = 1 - 1 / (self.t + self.warmup)
            beta = averaging_beta if self.beta2 == "avg" else self.beta2
            for _ in range(self.zsamples):
                z = 2 * sample_bernoulli(w.shape) - 1
                D += z * loss.hvp(w, z, i) / self.zsamples
            D = np.abs(beta * self.diagonal + (1 - beta) * D)
            D = np.maximum(D, self.alpha)
            self.diagonal = D
            precond_g = self.diagonal**-1 * g

        elif self.precond_type == "hutch++":
            # estimate hessian diagonal
            D = 0.0
            averaging_beta = 1 - 1 / (self.t + self.warmup)
            beta = averaging_beta if self.beta2 == "avg" else self.beta2
            m = self.zsamples
            S = np.random.randn(w.shape[0], m)
            G = np.random.randn(w.shape[0], m)
            hvp_S = loss.hvp(w, S, i)
            Q, _ = np.linalg.qr(hvp_S)
            hvp_Q = loss.hvp(w, Q, i)
            P = G - Q.dot(Q.T.dot(G))
            hvp_P = loss.hvp(w, P, i)
            H1 = Q * hvp_Q
            H2 = P * hvp_P
            D += H1.sum(1) + H2.sum(1) / m
            D = np.abs(beta * self.diagonal + (1 - beta) * D)
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
            m_corr = self.m / (1 - self.beta1 ** self.t)
            v_corr = self.v / (1 - self.beta2 ** self.t)
            self.diagonal = np.sqrt(v_corr) + self.alpha
            precond_g = self.diagonal**-1 * m_corr

        elif self.precond_type == "adadelta":
            self.v = self.beta2 * self.v + (1 - self.beta2) * g**2
            self.diagonal = np.sqrt(self.v + self.alpha) / np.sqrt(self.u + self.alpha)
            precond_g = self.diagonal**-1 * g
            self.u = self.beta2 * self.u + (1 - self.beta2) * precond_g**2

        elif self.precond_type == "adagrad":
            self.v += g**2
            self.diagonal = np.sqrt(self.v) + self.alpha
            precond_g = self.diagonal**-1 * g

        else:
            precond_g = g

        return precond_g


class SGD:
    def __init__(self, w, loss, BS=1, lr=0.0002, lr_decay=0, history_freq_per_epoch=5):
        self.w = w
        self.loss = loss
        self.N = self.loss.num_data  # @TODO: change N to num_data?
        self.BS = BS
        self.base_lr = lr
        # divide lr_decay by num updates per epoch to get approx `base_lr/(ep+1)`
        self.lr_decay = lr_decay / (self.N // self.BS)
        self.history_freq_per_epoch = history_freq_per_epoch
        self.reset_history()
        self.ep = 0  # effective passes over dataset
        self.t = 0  # num of updates/steps
        self.precond = None  # preconditioner

    def precondition(self, *args, **kwargs):
        self.precond = Preconditioner(*args, **kwargs)
        return self

    def precond_grad(self, g, i):
        if self.precond is not None:
            if self.precond.resample:
                i = np.random.choice(self.N, self.BS)
                self.ep += self.BS / self.N
            g = self.precond.update(self.w, self.loss, i, g)
        return g

    def init_run(self):
        # Initialize preconditioner
        if self.precond is not None:
            self.precond.init(self.w, self.loss, self.BS)
        # Record initial stats
        self.update_history()

    def run(self, T):
        # Run training loop
        self.init_run()
        for it in range(T * (self.N // self.BS)):
            self.step()
            if it % (self.N // (self.BS * self.history_freq_per_epoch)) == 0:
                self.update_history()

        return self.w, self.history

    def step(self):
        # Grad
        i = np.random.choice(self.N, self.BS)
        self.ep += self.BS / self.N
        g = self.loss.grad(self.w, i)

        precond_g = self.precond_grad(g, i)
        self.w -= self.lr * precond_g
        self.t += 1

        return g

    @property
    def lr(self):
        if self.lr_decay != 0:
            return self.base_lr / (1 + (self.t - 1) * self.lr_decay)
        else:
            return self.base_lr

    @property
    def history(self):
        return np.array(self._history)

    def reset_history(self):
        self._history = []

    def update_history(self):
        self._history.append(self.stats())

    def stats(self):
        # loss and gradient
        loss = self.loss.func(self.w)
        g_norm = np.linalg.norm(self.loss.grad(self.w))**2

        # Error
        prediction = self.loss.pred(self.w)
        error = np.mean(prediction < 0)  # wrong prediction -> 100% error
        error += 0.5 * np.mean(prediction == 0)  # ambiguous prediction -> 50% error

        # Preconditioner statistics @TODO: when should we report this?
        D_ratio = 0.
        if self.precond is not None:
            D_ratio = np.mean(self.precond.diagonal > self.precond.alpha)

        H_diag_err = 0.
        if self.precond == "hutchinson":
            # H_diag = self.loss.hessian_diag(self.w)
            # H_diag_err = np.linalg.norm(self.precond.diagonal - H_diag) / np.linalg.norm(H_diag)
            pass

        return (self.ep, loss, g_norm, error, D_ratio, H_diag_err)


class SVRG(SGD):
    def __init__(self, w, loss, inner_loop=1.0, **kwargs):
        super().__init__(w, loss, **kwargs)
        # Size of inner loop as a multiple of the dataset length
        self.inner_loop = inner_loop
        # Outer loop / checkpoint weights
        self.w_out = np.array(self.w)
        # Estimate of full batch gradient on outer loop weights
        self.g_full = None

    def run(self, T):
        # Run training loop
        self.init_run()
        for epoch in range(T):
            # Update full batch gradient
            self.g_full = self.loss.grad(self.w_out)
            self.ep += 1
            gradnorm0 = np.linalg.norm(self.g_full)

            for it in range(10**10):
                g = self.step()

                # Inner loop stopping criterion
                if np.linalg.norm(g) < 0.1 * gradnorm0 \
                        or it > (self.N // self.BS) * self.inner_loop:
                    self.w_out[:] = self.w[:]
                    self.update_history()
                    break

                if it % (self.N // (self.BS * self.history_freq_per_epoch)) == 0:
                    self.update_history()

        return self.w, self.history

    def step(self):
        # Grad
        i = np.random.choice(self.N, self.BS)
        self.ep += self.BS / self.N
        g_in = self.loss.grad(self.w, i)
        g_out = self.loss.grad(self.w_out, i)
        g = self.g_full + g_in - g_out

        # Update
        precond_g = self.precond_grad(g, i)
        self.w -= self.lr * precond_g
        self.t += 1

        return g


class SARAH(SVRG):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self):
        # Grad
        i = np.random.choice(self.N, self.BS)
        self.ep += self.BS / self.N
        g_in = self.loss.grad(self.w, i)
        g_out = self.loss.grad(self.w_out, i)
        self.g_full += g_in - g_out
        g = self.g_full

        # Update
        self.w_out[:] = self.w[:]
        precond_g = self.precond_grad(g, i)
        self.w -= self.lr * precond_g
        self.t += 1

        return g


class LSVRG(SVRG):
    def __init__(self, w, loss, p=0.99, **kwargs):
        super().__init__(w, loss, **kwargs)
        self.p = p

    def run(self, T):
        self.init_run()

        # Update full batch gradient
        self.g_full = self.loss.grad(self.w_out)
        self.ep += 1

        # Run training loop
        for it in range(T * (self.N // self.BS)):
            self.step()

            # Checkpoint criterion (instead of inner loop stopping criterion)
            if sample_uniform() > self.p:
                self.w_out[:] = self.w[:]
                self.g_full = self.loss.grad(self.w_out)
                self.ep += 1

            if it % (self.N // (self.BS * self.history_freq_per_epoch)) == 0:
                self.update_history()

        return self.w, self.history


class PAGE(SGD):
    # @TODO: option for g_batch
    def __init__(self, w, loss, p=0.99, g_batch=None, **kwargs):
        super().__init__(w, loss, **kwargs)
        # PAGE generalizes SARAH and SGD
        # p is opposite to the definition, just to make it more similar to L-SVRG
        # g_batch is the size of the batch on which g_full is estimated
        self.p = p
        self.g_batch = g_batch
        # Checkpoint weights
        self.w_out = np.array(self.w)
        # Estimate of full batch gradient on outer loop weights
        self.g_full = None

    def update_full_grad(self):
        if self.g_batch is None:
            i = None
            self.ep += 1
        else:
            i = np.random.choice(self.N, self.g_batch)
            self.ep += self.BS / self.N
        self.g_full = self.loss.grad(self.w_out, i)

        return self.g_full

    def init_run(self):
        super().init_run()
        # Update full batch gradient
        self.update_full_grad()

    def step(self):
        i = np.random.choice(self.N, self.BS)
        self.ep += self.BS / self.N

        # Grad
        if self.p == 1 or sample_uniform() < self.p:
            # SARAH
            g_in = self.loss.grad(self.w, i)
            g_out = self.loss.grad(self.w_out, i)
            self.g_full += g_in - g_out
            g = self.g_full
        else:
            # GD
            g = self.update_full_grad()

        # Update
        self.w_out[:] = self.w[:]
        precond_g = self.precond_grad(g, i)
        self.w -= self.lr * precond_g
        self.t += 1

        return g


class SuperSGD(SGD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reg_pow = 0.9
        self.reg_const = 1.0
        self.reg_const_min = 1e-12
        self.search_max_iter = 2

    def step(self):
        for j in range(10**10):
            # Grad
            i = np.random.choice(self.N, self.BS)
            self.ep += self.BS / self.N
            g = self.loss.grad(self.w, i)
            gnorm = np.linalg.norm(g)
            # Regularize
            self.precond.alpha = self.reg_const * gnorm**self.reg_pow
            precond_g = self.precond_grad(g, i)
            # Update
            w_next = self.w - self.lr * precond_g
            if j + 1 == self.search_max_iter:
                # print(f"Exceeded max backtracking iterations (reg = {self.precond.alpha}).")
                break
            # Backtrack
            g_next = self.loss.grad(w_next, i)
            gnorm_next = np.linalg.norm(g_next)
            # Check stopping criterion
            if g_next.dot(self.w - w_next) >= gnorm_next**2 / (4 * self.precond.alpha):
                self.reg_const = max(0.25 * self.reg_const, self.reg_const_min)
                break
            self.reg_const *= 4

        self.w = w_next
        self.t += 1

        return g


class SuperLSVRG(LSVRG):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reg_pow = 0.9
        self.reg_const = 1.0
        self.reg_const_min = 1e-12
        self.search_max_iter = 40

    def step(self):
        for j in range(10**10):
            # Grad
            i = np.random.choice(self.N, self.BS)
            self.ep += self.BS / self.N
            g_in = self.loss.grad(self.w, i)
            g_out = self.loss.grad(self.w_out, i)
            g = self.g_full + g_in - g_out
            gnorm = np.linalg.norm(g)
            # Regularize
            self.precond.alpha = self.reg_const * gnorm**self.reg_pow
            precond_g = self.precond_grad(g, i)
            # Update
            w_next = self.w - self.lr * precond_g
            if j + 1 == self.search_max_iter:
                # print(f"Exceeded max backtracking iterations (reg = {self.precond.alpha}).")
                break
            # Backtrack
            g_next = self.g_full + self.loss.grad(w_next, i) - g_out
            gnorm_next = np.linalg.norm(g_next)
            # Check stopping criterion
            if g_next.dot(self.w - w_next) >= gnorm_next**2 / (4 * self.precond.alpha):
                self.reg_const = max(0.25 * self.reg_const, self.reg_const_min)
                break
            self.reg_const *= 4

        self.w = w_next
        self.t += 1

        return g


class SuperSARAH(SARAH):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reg_pow = 0.9
        self.reg_const = 1.0
        self.reg_const_min = 1e-12
        self.search_max_iter = 40

    def step(self):
        for j in range(10**10):
            # Grad
            i = np.random.choice(self.N, self.BS)
            self.ep += self.BS / self.N
            g_in = self.loss.grad(self.w, i)
            g_out = self.loss.grad(self.w_out, i)
            g = self.g_full + g_in - g_out
            gnorm = np.linalg.norm(g)
            # Regularize
            self.precond.alpha = self.reg_const * gnorm**self.reg_pow
            precond_g = self.precond_grad(g, i)
            # Update
            w_next = self.w - self.lr * precond_g
            if j + 1 == self.search_max_iter:
                print(f"Exceeded max backtracking iterations (reg = {self.precond.alpha}).")
                break
            # Backtrack
            g_next = self.g_full + self.loss.grad(w_next, i) - g_in
            gnorm_next = np.linalg.norm(g_next)
            # Check stopping criterion
            if g_next.dot(self.w - w_next) >= gnorm_next**2 / (4 * self.precond.alpha):
                self.reg_const = max(0.25 * self.reg_const, self.reg_const_min)
                break
            self.reg_const *= 4

        self.g_full = g
        self.w_out[:] = self.w[:]
        self.w = w_next
        self.t += 1

        return g


class Adam(SGD):
    def __init__(self, w, loss, BS=1, lr=0.001, lr_decay=0, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(w, loss, BS=BS, lr=lr, lr_decay=lr_decay)
        self.precondition("adam", beta1=beta1, beta2=beta2, alpha=eps)


class Adagrad(SGD):
    def __init__(self, w, loss, BS=1, lr=0.01, lr_decay=0, eps=1e-10, *args, **kwargs):
        super().__init__(w, loss, BS=BS, lr=lr, lr_decay=lr_decay)
        self.precondition("adagrad", alpha=eps)


class Adadelta(SGD):
    def __init__(self, w, loss, BS=1, lr=1.0, lr_decay=0, rho=0.9, eps=1e-6, *args, **kwargs):
        super().__init__(w, loss, BS=BS, lr=lr, lr_decay=lr_decay)
        self.precondition("adadelta", beta2=rho, alpha=eps)


###############################################################################
# Utility functions for running experiments @TODO: do we need this?

def run_SGD(X, y, w, loss, T=10000, BS=1, lr=0.2, lr_decay=0, weight_decay=0, **precond_args):
    optim = SGD(w, loss, BS=BS, lr=lr, lr_decay=lr_decay)
    optim = optim.precondition(**precond_args)
    return optim.run(T)


def run_SVRG(X, y, w, loss, T=10000, BS=1, lr=0.2, lr_decay=0, weight_decay=0, inner_loop=1.0, **precond_args):
    optim = SVRG(w, loss, BS=BS, lr=lr, lr_decay=lr_decay, inner_loop=inner_loop)
    optim = optim.precondition(**precond_args)
    return optim.run(T)


def run_LSVRG(X, y, w, loss, T=10000, BS=1, lr=0.2, lr_decay=0, weight_decay=0, p=0.99, **precond_args):
    optim = LSVRG(w, loss, BS=BS, lr=lr, lr_decay=lr_decay, p=p)
    optim = optim.precondition(**precond_args)
    return optim.run(T)


def run_PAGE(X, y, w, loss, T=10000, BS=1, lr=0.2, lr_decay=0, weight_decay=0, p=0.99, **precond_args):
    optim = PAGE(w, loss, BS=BS, lr=lr, lr_decay=lr_decay, p=p)
    optim = optim.precondition(**precond_args)
    return optim.run(T)


def run_SARAH(X, y, w, loss, T=10000, BS=1, lr=0.2, lr_decay=0, weight_decay=0, **precond_args):
    optim = SARAH(w, loss, BS=BS, lr=lr, lr_decay=lr_decay)
    optim = optim.precondition(**precond_args)
    return optim.run(T)


# -------------- SUPER -------------- #
def run_SuperSGD(X, y, w, loss, T=10000, BS=1, lr=0.2, lr_decay=0, weight_decay=0, **precond_args):
    optim = SuperSGD(w, loss, BS=BS, lr=lr, lr_decay=lr_decay)
    optim = optim.precondition(**precond_args)
    return optim.run(T)


def run_SuperLSVRG(X, y, w, loss, T=10000, BS=1, lr=0.2, lr_decay=0, weight_decay=0, p=0.99, **precond_args):
    p = 1 - 1 / loss.num_data**0.5 if p == 'auto' else float(p)
    print(f"p = {p}")
    optim = SuperLSVRG(w, loss, BS=BS, lr=lr, lr_decay=lr_decay, p=p)
    optim = optim.precondition(**precond_args)
    return optim.run(T)


def run_SuperSARAH(X, y, w, loss, T=10000, BS=1, lr=0.2, lr_decay=0, weight_decay=0, **precond_args):
    optim = SuperSARAH(w, loss, BS=BS, lr=lr, lr_decay=lr_decay)
    optim = optim.precondition(**precond_args)
    return optim.run(T)
# ----------------------------------- #


def run_Adam(X, y, w, loss, T=10000, BS=1, lr=0.2, lr_decay=0, weight_decay=0, beta1=0.9, beta2=0.999, alpha=1e-8, **_):
    optim = Adam(w, loss, BS=BS, lr=lr, lr_decay=lr_decay, beta1=beta1, beta2=beta2, eps=alpha)
    return optim.run(T)


def run_Adagrad(X, y, w, loss, T=10000, BS=1, lr=0.2, lr_decay=0, weight_decay=0, alpha=1e-10, **_):
    optim = Adagrad(w, loss, BS=BS, lr=lr, lr_decay=lr_decay, eps=alpha)
    return optim.run(T)


def run_Adadelta(X, y, w, loss, T=10000, BS=1, lr=1.0, lr_decay=0, weight_decay=0, beta2=0.9, alpha=1e-6, **_):
    optim = Adadelta(w, loss, BS=BS, lr=lr, lr_decay=lr_decay, rho=beta2, eps=alpha)
    return optim.run(T)
