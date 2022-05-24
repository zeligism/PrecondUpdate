
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ScaledSVRG(optim.Optimizer):
    def __init__(self, params, lr=0.02, period=10**10,
                 beta=0.999, alpha=1e-5, warmup=100, SARAH=False):
        self.SARAH = SARAH  # change SVRG to SARAH
        defaults = dict(lr=lr, beta=beta, alpha=alpha)
        super().__init__(params, defaults)
        self.global_state.setdefault('t', 0)  # num of step iters
        self.global_state.setdefault('ckpt_evals', 0)  # num of checkpoints
        self.global_state.setdefault('should_ckpt', True)  # we should checkpoint initially
        self.global_state.setdefault('ckpt_period', period)  # period of checkpointing (in t)
        self.global_state.setdefault('warmup', warmup)  # num of diagonal warmup iters

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

        gz_sum = 0.
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                z = 2 * torch.randint_like(p.grad, 2) - 1
                gz_sum += (p.grad * z).sum()
                self.state[p]['z'] = z
        gz_sum.backward()
        with torch.no_grad():
            for group in self.param_groups:
                beta = group['beta']
                alpha = group['alpha']
                if beta == "avg":
                    beta = 1 - 1 / (t + warmup)
                for p in group['params']:
                    if p.grad is None:
                        continue
                    if 'D' not in self.state[p]:
                        self.state[p]['D'] = 0.
                    # D = z * p.grad = z * d(df(w)/dw z)/dw = z * d^2f(w)/d^2w z = z * H z
                    D = self.state[p]['z'] * p.grad
                    self.state[p]['z'] = None
                    if init_phase:
                        self.state[p]['D'] += D / warmup
                        # TODO: is this necessary?
                        if warmup == t-1:
                            D = self.state[p]['D'].abs()
                            D[D < alpha] = alpha
                            self.state[p]['D'] = D
                    else:
                        D_prev = self.state[p]['D']
                        D = (beta * D_prev + (1 - beta) * D).abs()
                        D[D < alpha] = alpha
                        self.state[p]['D'] = D

    @torch.no_grad()
    def update_ckpt(self):
        """
        Checkpoint params and full grad.
        """
        ckpt_gradnorm = 0.
        for group in self.param_groups:
            for p in group['params']:
                pstate = self.state[p]
                pstate['ckpt'] = p.detach().clone()
                if p.grad is None:
                    pstate['full_grad'] = torch.zeros_like(pstate['ckpt'])
                else:
                    pstate['full_grad'] = p.grad.detach().clone()
                    ckpt_gradnorm += torch.sum(p.grad**2)
        self.global_state['ckpt_gradnorm'] = ckpt_gradnorm.sqrt().item()
        self.global_state['ckpt_evals'] += 1
        self.global_state['should_ckpt'] = False

    @torch.no_grad()
    def step(self, closure):
        t = self.global_state['t']
        warmup = self.global_state['warmup']
        ckpt_period = self.global_state['ckpt_period']
        should_ckpt = self.global_state['should_ckpt'] or (t % ckpt_period) == 0

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

        ##### Update checkpoint params and full grad #####
        if should_ckpt:
            print(f"Checkpointing...")
            closure(full_batch=True)
            self.update_ckpt()

        ##### Update diagonal #####
        closure(create_graph=True)
        self.update_diagonal()

        ##### Take step #####
        gradnorm = 0.
        # Gather stochastic grads on orig params
        loss = closure()
        # Store orig params and grads, and set params to ckpt params
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['orig'] = p.detach().clone()
                self.state[p]['orig_grad'] = p.grad.detach().clone()
                p.set_(self.state[p]['ckpt'].data)
        # Gather stochastic grads of loss on ckpt params
        closure()
        # Reset params and update grads
        for group in self.param_groups:
            for p in group['params']:
                orig_param = self.state[p]['orig']
                ckpt_grad = p.grad if p.grad is not None else 0.
                orig_grad = self.state[p]['orig_grad']
                full_grad = self.state[p]['full_grad']
                if orig_grad is None:
                    self.state[p]['orig'] = None
                    continue
                grad = full_grad - ckpt_grad + orig_grad
                ### Add this line for SVRG -> SARAH ###
                if self.SARAH:
                    self.state[p]['ckpt'] = orig_param.detach().clone()
                    self.state[p]['full_grad'] = grad.detach().clone()
                gradnorm += torch.sum(grad**2)
                # precondition grad and take a step
                if 'D' in self.state[p]:
                    grad.mul_(self.state[p]['D']**-1)
                p.set_(orig_param - group['lr'] * grad)
                p.grad.detach_()
                p.grad.zero_()
                self.state[p]['orig'] = None
                self.state[p]['orig_grad'] = None
        gradnorm = gradnorm.sqrt().item()

        self.global_state['should_ckpt'] = gradnorm < 0.1 * self.global_state['ckpt_gradnorm']
        self.global_state['t'] += 1
        return loss


class ScaledSARAH(ScaledSVRG):
    def __init__(self, params, **kwargs):
        kwargs['SARAH'] = True
        super().__init__(params, **kwargs)


class ScaledLSVRG(ScaledSVRG):
    def __init__(self, params, p=0.99, **kwargs):
        if 'ckpt_period' in kwargs:
            del kwargs['ckpt_period']
        super().__init__(params, **kwargs)
        self.global_state.setdefault('ckpt_prob', p)  # prob of checkpointing
        self.global_state.setdefault('ckpt_period', 10**10)  # don't use period

    @torch.no_grad()
    def step(self, closure):
        # Should checkpoint with prob `ckpt_prob`
        should_ckpt = torch.rand(1).item() > self.global_state['ckpt_prob']
        self.global_state['should_ckpt'] = self.global_state['should_ckpt'] or should_ckpt
        return super().step(closure)


### TODO: add ScaledSGD?

