
import torch
import torch.optim as optim
from .vr import *


class ScaledOptimizer(optim.Optimizer):
    def init_precond(self, warmup=100, beta=0.999, alpha=1e-5, zsamples=1, layer_wise=True, scaled_z=True):
        for group in self.param_groups:
            group.setdefault('beta', beta)
            group.setdefault('alpha', alpha)
        self.global_state.setdefault('zsamples', zsamples)  # num of z samples in update_diagonal
        self.global_state.setdefault('warmup', warmup)  # num of diagonal warmup iters
        self.global_state.setdefault('D_iters', 0)  # num of diagonal updates
        self.global_state.setdefault('layer_wise', layer_wise)
        self.global_state.setdefault('scaled_z', scaled_z)

    def __setstate__(self, state):
        self.__setstate__(state)
        for group in self.param_groups:
            group.setdefault('beta', 0.999)
            group.setdefault('alpha', 1e-5)
        self.global_state.setdefault('zsamples', 1)  # num of z samples in update_diagonal
        self.global_state.setdefault('warmup', 1)  # num of diagonal warmup iters
        self.global_state.setdefault('D_iters', 0)  # num of diagonal updates
        self.global_state.setdefault('layer_wise', True)
        self.global_state.setdefault('scaled_z', True)

    @property
    def global_state(self):
        # First param holds global state
        p0 = self.param_groups[0]['params'][0]
        return self.state[p0]

    @torch.no_grad()
    def update_diagonal(self, init_phase=False):
        """
        Updates diagonal based on Hutchinson trace estimation.
        `closure(create_graph=True)` should be called right before calling this method.
        """
        if self.global_state['layer_wise']:
            return self.update_diagonal_layerwise(init_phase=init_phase)
        else:
            return self.update_diagonal_fullmodel(init_phase=init_phase)

    # ---------- Full model implementation ---------- #
    @torch.no_grad()
    def update_diagonal_fullmodel(self, init_phase=False):
        D_iters = self.global_state['D_iters']

        gz_sum = 0.  # its derivative is the hessian-vector product Hz
        for group in self.param_groups:
            for p in group['params']:
                pstate = self.state[p]
                if p.grad is None:
                    continue
                z = 2 * torch.randint_like(p, 2) - 1
                with torch.enable_grad():
                    gz_sum += (p.grad * z).sum()
                pstate['z'] = z.clone().detach()
        with torch.enable_grad():
            gz_sum.backward()

        # Apply scaled update
        for group in self.param_groups:
            use_avg_beta = init_phase or group['beta'] in ("avg", "auto")
            beta = 1 - 1 / (D_iters + 1) if use_avg_beta else group['beta']
            for p in group['params']:
                pstate = self.state[p]
                if p.grad is None:
                    continue
                # Hutchinson: D = z * p.grad = z * d(df(w)/dw z)/dw = z * d^2f(w)/d^2w z = z * H z
                D = pstate['z'].mul(p.grad)  # recall D = z o Hz, and p.grad holds Hz
                pstate['z'] = None
                # update diagonal
                if 'D' not in pstate:
                    pstate['D'] = D
                pstate['D'].mul_(beta).add_(D.mul(1 - beta))
                p.grad = None

        self.global_state['D_iters'] += 1

    # ---------- Layer-wise implementation ---------- #
    @torch.no_grad()
    def update_diagonal_layerwise(self, init_phase=False):
        D_iters = self.global_state['D_iters']
        scaled_z = self.global_state['scaled_z']
        zsamples = self.global_state['zsamples'] if init_phase else 1

        for group in self.param_groups:
            use_avg_beta = init_phase or group['beta'] in ("avg", "auto")
            beta = 1 - 1 / (D_iters + 1) if use_avg_beta else group['beta']
            alpha = group["alpha"]
            for p in group['params']:
                if p.grad is None:
                    continue
                pstate = self.state[p]
                D = torch.zeros_like(p)
                for _ in range(zsamples):
                    z = torch.randn_like(p)  # XXX
                    scale = 1
                    if scaled_z and 'D' in pstate:
                        scale = pstate['D'].abs().clamp(min=alpha).sqrt()
                    with torch.enable_grad():
                        Hz, = torch.autograd.grad(p.grad, p, grad_outputs=z.div(scale), retain_graph=True)
                    D.add_(z.mul(scale) * Hz / zsamples)
                # update diagonal
                if 'D' not in pstate:
                    pstate['D'] = D
                pstate['D'].mul_(beta).add_(D.mul(1 - beta))
                p.grad = None

        self.global_state['D_iters'] += 1

    @torch.no_grad()
    def step(self, closure):
        D_iters = self.global_state['D_iters']
        warmup = self.global_state['warmup']

        closure = torch.enable_grad()(closure)  # always enable grad for closure

        ##### Warming up diagonal #####
        if D_iters < warmup:
            if D_iters == 0:
                print("Warming up...")
            loss = closure(create_graph=True)
            self.update_diagonal(init_phase=True)
            return loss

        ##### Update diagonal #####
        elif D_iters == warmup:
            print("Warm up done.")

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
                    alpha = group["alpha"]
                    D = pstate['D'].abs().clamp(min=alpha)
                    p.copy_(pstate['orig'].addcdiv(pstate['orig'].sub(p), D, value=-1))

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



