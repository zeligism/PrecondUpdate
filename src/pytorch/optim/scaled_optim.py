
import torch
import torch.optim as optim
from .vr import *


class ScaledOptimizer(optim.Optimizer):
    def init_precond(self, warmup=100, beta=0.999, alpha=1e-5,
                     reg_const=1.0, reg_const_min=1e-5, reg_pow=1.0):
        # TODO: just set all as global state
        for group in self.param_groups:
            group.setdefault('beta', beta)
            group.setdefault('alpha', alpha)
            group.setdefault('super', alpha in ('auto', 'super'))
            group.setdefault('reg_const', reg_const)
            group.setdefault('reg_const_min', reg_const_min)
            group.setdefault('reg_pow', reg_pow)
        self.global_state.setdefault('warmup', warmup)  # num of diagonal warmup iters
        self.global_state.setdefault('t', 0)  # num of step iters

    def __setstate__(self, state):
        self.__setstate__(state)
        for group in self.param_groups:
            group.setdefault('beta', 0.999)
            group.setdefault('alpha', 1e-5)
            group.setdefault('super', False)
            group.setdefault('reg_const', 1.0)
            group.setdefault('reg_const_min', 1e-5)
            group.setdefault('reg_pow', 0.9)
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
            if group['super']:
                if 'alpha_hist' not in group:
                    group['alpha_hist'] = []
                group['alpha'] = group['reg_const'] * gradnorm**group['reg_pow']
                group['alpha_hist'].append(group['alpha'])
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
                # precondition step
                delta = pstate['orig'] - p
                effective_grad = delta / group['lr']
                if 'D' in self.state[p]:
                    p.data.set_(pstate['orig'] - pstate['D']**-1 * delta)
                # for checking backtracking condition
                dot += torch.sum(effective_grad * (pstate['orig'] - p)).item()
                gradnorm_sq += torch.sum(effective_grad**2).item()
            if 4 * group['alpha'] * dot >= gradnorm_sq:
                group['reg_const'] = max(0.25 * group['reg_const'], group['reg_const_min'])
            else:
                group['reg_const'] = 4 * group['reg_const']

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



