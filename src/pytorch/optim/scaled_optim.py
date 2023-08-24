
import torch
import torch.optim as optim
from .vr import *


class ScaledOptimizer(optim.Optimizer):
    def init_precond(self, warmup=100, beta=0.999, alpha=1e-5, zsamples=1):
        for group in self.param_groups:
            group.setdefault('beta', beta)
            group.setdefault('alpha', alpha)
        self.global_state.setdefault('zsamples', zsamples)  # num of z samples in update_diagonal
        self.global_state.setdefault('warmup', warmup)  # num of diagonal warmup iters
        self.global_state.setdefault('t', 0)  # num of step iters

    def __setstate__(self, state):
        self.__setstate__(state)
        for group in self.param_groups:
            group.setdefault('beta', 0.999)
            group.setdefault('alpha', 1e-5)
        self.global_state.setdefault('zsamples', 1)  # num of z samples in update_diagonal
        self.global_state.setdefault('warmup', 1)  # num of diagonal warmup iters
        self.global_state.setdefault('t', 0)  # num of step iters

    @property
    def global_state(self):
        # First param holds global state
        p0 = self.param_groups[0]['params'][0]
        return self.state[p0]

    @torch.no_grad()
    def update_diagonal(self, init_phase=False, layer_wise=True):
        """
        Updates diagonal based on Hutchinson trace estimation.
        `closure(create_graph=True)` should be called right before calling this method.
        """
        t = self.global_state['t']
        warmup = self.global_state['warmup']
        D_diff = 0.
        D_sum = 0.

        # ---------- Layer-wise implementation ---------- #
        if layer_wise:
            zsamples = self.global_state['zsamples'] if init_phase else 1
            with torch.enable_grad():
                for group in self.param_groups:
                    for p in group['params']:
                        pstate = self.state[p]
                        D = torch.zeros_like(p)
                        for _ in range(zsamples):
                            z = torch.randint_like(p, 2) * 2 - 1
                            hv, = torch.autograd.grad(p.grad, p, grad_outputs=z, retain_graph=True)
                            D.add_(z * hv / zsamples)
                        if 'D_t' in pstate:
                            D_diff += torch.norm(pstate['D_t'] - D)**2
                        pstate['D_t'] = D
        
        # ---------- Full model implementation ---------- #
        else:
            with torch.enable_grad():
                gz_sum = 0.  # its derivative is the hessian-vector product Hz
                for group in self.param_groups:
                    for p in group['params']:
                        pstate = self.state[p]
                        if p.grad is None:
                            continue
                        z = 2 * torch.randint_like(p, 2) - 1
                        gz_sum += (p.grad * z).sum()
                        pstate['z'] = z.clone().detach()
                gz_sum.backward()

        # -------------------- #

        # Apply scaled update
        for group in self.param_groups:
            beta = 1 - 1 / (t + warmup) if group['beta'] in ("avg", "auto") else group['beta']
            for p in group['params']:
                pstate = self.state[p]
                if p.grad is None:
                    continue
                if layer_wise:
                    D = pstate['D_t']
                else:
                    # Hutchinson: D = z * p.grad = z * d(df(w)/dw z)/dw = z * d^2f(w)/d^2w z = z * H z
                    D = pstate['z'].mul(p.grad)  # recall D = z o Hz, and p.grad holds Hz
                    if 'D_t' in pstate:
                        D_diff += torch.norm(pstate['D_t'] - D)**2
                    pstate['D_t'] = D
                    pstate['z'] = None
                # update diagonal
                if init_phase:
                    if 'D' not in pstate:
                        pstate['D'] = D.div(warmup)
                    else:
                        pstate['D'].add_(D.div(warmup))
                else:
                    pstate['D'] = pstate['D'].mul(beta).add(D.mul(1 - beta))
                D_sum += torch.norm(pstate['D'])**2
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
            alpha = group['alpha']
            for p in group['params']:
                # precondition step
                if 'D' in self.state[p]:
                    pstate = self.state[p]
                    delta = pstate['orig'].sub(p)
                    D = pstate['D'].abs().clamp(min=alpha)
                    p.copy_(pstate['orig'].addcdiv(delta, D, value=-1))

        self.global_state['t'] += 1
        return loss


class ScaledSGD(ScaledOptimizer, optim.SGD):
    def __init__(self, params, warmup=100, zsamples=1, beta=0.999, alpha=1e-5, **optim_args):
        super().__init__(params, **optim_args)
        self.init_precond(warmup=warmup, zsamples=zsamples, beta=beta, alpha=alpha)


class ScaledSVRG(ScaledOptimizer, SVRG):
    def __init__(self, params, warmup=100, zsamples=1, beta=0.999, alpha=1e-5, **optim_args):
        super().__init__(params, **optim_args)
        self.init_precond(warmup=warmup, zsamples=zsamples, beta=beta, alpha=alpha)


class ScaledLSVRG(ScaledOptimizer, LSVRG):
    def __init__(self, params, warmup=100, zsamples=1, beta=0.999, alpha=1e-5, **optim_args):
        super().__init__(params, **optim_args)
        self.init_precond(warmup=warmup, zsamples=zsamples, beta=beta, alpha=alpha)


class ScaledSARAH(ScaledOptimizer, SARAH):
    def __init__(self, params, warmup=100, zsamples=1, beta=0.999, alpha=1e-5, **optim_args):
        super().__init__(params, **optim_args)
        self.init_precond(warmup=warmup, zsamples=zsamples, beta=beta, alpha=alpha)



