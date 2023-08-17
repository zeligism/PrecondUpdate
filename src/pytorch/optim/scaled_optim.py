
import torch
import torch.optim as optim
from .vr import *


class ScaledOptimizer(optim.Optimizer):
    def init_precond(self, warmup=100, beta=0.999, alpha=1e-5):
        # TODO: just set all as global state
        for group in self.param_groups:
            group.setdefault('beta', beta)
            group.setdefault('alpha', alpha)
        self.global_state.setdefault('warmup', warmup)  # num of diagonal warmup iters
        self.global_state.setdefault('t', 0)  # num of step iters

    def __setstate__(self, state):
        self.__setstate__(state)
        for group in self.param_groups:
            group.setdefault('beta', 0.999)
            group.setdefault('alpha', 1e-5)
        self.global_state.setdefault('warmup', 1)  # num of diagonal warmup iters
        self.global_state.setdefault('t', 0)  # num of step iters

    @property
    def global_state(self):
        # First param holds global state
        p0 = self.param_groups[0]['params'][0]
        return self.state[p0]

    @torch.enable_grad()
    def update_diagonal(self, init_phase=False):
        """
        Updates diagonal based on Hutchinson trace estimation.
        `closure(create_graph=True)` should be called right before calling this method.
        """
        t = self.global_state['t']
        warmup = self.global_state['warmup']

        gz_sum = 0.  # hessian-vector product
        for group in self.param_groups:
            gradnorm = 0.
            for p in group['params']:
                pstate = self.state[p]
                if p.grad is None:
                    continue
                # z = 2 * torch.randint_like(p.grad, 2) - 1
                # gz_sum += (p.grad * z).sum()
                # pstate['z'] = z
                # gradnorm += torch.sum(p.grad**2).item()
                c = 0.0  # TODO: add momentum?
                if 'grad_avg' not in pstate:
                    pstate['grad_avg'] = p.grad.detach()
                grad = c * pstate['grad_avg'] + (1 - c) * p.grad
                z = 2 * torch.randint_like(grad, 2) - 1
                gz_sum += (grad * z).sum()
                pstate['z'] = z
                gradnorm += torch.sum(grad**2).item()
                pstate['grad_avg'] = grad.detach()
            gradnorm = gradnorm**0.5
        gz_sum.backward()

        with torch.no_grad():
            for group in self.param_groups:
                alpha = group['alpha']
                beta = 1 - 1 / (t + warmup) if group['beta'] in ("avg", "auto") else group['beta']
                for p in group['params']:
                    pstate = self.state[p]
                    if p.grad is None:
                        continue
                    # Hutchinson: D = z * p.grad = z * d(df(w)/dw z)/dw = z * d^2f(w)/d^2w z = z * H z
                    D = pstate['z'] * p.grad
                    pstate['z'] = None
                    # update diagonal
                    if init_phase:
                        if 'D' not in pstate:
                            pstate['D'] = 0.
                        pstate['D'] += D / warmup
                        # TODO: is this necessary?
                        if t - 1 == warmup:
                            D = pstate['D'].abs()
                            D[D < alpha] = alpha
                            pstate['D'] = D
                    else:
                        D_prev = pstate['D']
                        D = (beta * D_prev + (1 - beta) * D).abs()
                        D[D < alpha] = alpha
                        pstate['D'] = D
                    p.grad = None

    @torch.no_grad()
    def step(self, closure):
        t = self.global_state['t']
        warmup = self.global_state['warmup']

        closure = torch.enable_grad()(closure)  # always enable grad for closure

        ##### Warming up diagonal #####
        if t < warmup:
            if t == 0:
                print("Warming up...")
            loss = closure(create_graph=True)
            self.update_diagonal(init_phase=True)
            self.global_state['t'] += 1
            return loss
        elif t == warmup:
            print("Warm up done.")

        ##### Update diagonal #####
        closure(create_graph=True)
        self.update_diagonal()

        ##### Take step #####
        # Store original params
        for group in self.param_groups:
            for p in group['params']:
                pstate = self.state[p]
                pstate['orig'] = p.detach().clone()

        # Take optimizer's step
        loss = super().step(closure)

        # Get the step and reapply it with preconditioning
        for group in self.param_groups:
            gradnorm_sq = 0.
            dot = 0.
            for p in group['params']:
                pstate = self.state[p]
                delta = pstate['orig'] - p
                # for checking backtracking condition
                if 'prev_delta' in pstate:
                    grad = delta / group['lr']  # the effective grad
                    dot += torch.sum(grad * pstate['prev_delta']).item()
                    gradnorm_sq += torch.sum(grad**2).item()
                # precondition step
                pstate['prev_delta'] = delta.detach().clone()
                if 'D' in self.state[p]:
                    p.copy_(pstate['orig'] - delta / pstate['D'])

        self.global_state['t'] += 1
        return loss


class ScaledSGD(ScaledOptimizer, optim.SGD):
    def __init__(self, params, warmup=100, beta=0.999, alpha=1e-5, **optim_args):
        super().__init__(params, **optim_args)
        self.init_precond(warmup=warmup, beta=beta, alpha=alpha)


class ScaledSVRG(ScaledOptimizer, SVRG):
    def __init__(self, params, warmup=100, beta=0.999, alpha=1e-5, **optim_args):
        super().__init__(params, **optim_args)
        self.init_precond(warmup=warmup, beta=beta, alpha=alpha)


class ScaledLSVRG(ScaledOptimizer, LSVRG):
    def __init__(self, params, warmup=100, beta=0.999, alpha=1e-5, **optim_args):
        super().__init__(params, **optim_args)
        self.init_precond(warmup=warmup, beta=beta, alpha=alpha)


class ScaledSARAH(ScaledOptimizer, SARAH):
    def __init__(self, params, warmup=100, beta=0.999, alpha=1e-5, **optim_args):
        super().__init__(params, **optim_args)
        self.init_precond(warmup=warmup, beta=beta, alpha=alpha)



