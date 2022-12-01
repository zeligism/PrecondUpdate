
import torch
import torch.optim as optim


class ScaledSGD(optim.Optimizer):
    def __init__(self, params, lr=0.02, beta=0.999, alpha=1e-5, warmup=100,
                 reg_const=1.0, reg_const_min=1e-5, reg_pow=1.0):
        defaults = dict(lr=lr, beta=beta, alpha=alpha)
        defaults = dict(**defaults, reg_const=reg_const, reg_const_min=reg_const_min, reg_pow=reg_pow)
        super().__init__(params, defaults)
        self.global_state.setdefault('t', 0)  # num of step iters
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

        gz_sum = 0.  # hessian-vector product
        for group in self.param_groups:
            gradnorm_sq = 0.
            for p in group['params']:
                if p.grad is None:
                    continue
                z = 2 * torch.randint_like(p.grad, 2) - 1
                gz_sum += (p.grad * z).sum()
                self.state[p]['z'] = z
                gradnorm_sq += torch.sum(p.grad**2).item()
            group['alpha'] = group['reg_const'] * gradnorm_sq**(0.5 * group['reg_const'])
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
                        if warmup == t - 1:
                            D = self.state[p]['D'].abs()
                            D[D < alpha] = alpha
                            self.state[p]['D'] = D
                    else:
                        D_prev = self.state[p]['D']
                        D = (beta * D_prev + (1 - beta) * D).abs()
                        D[D < alpha] = alpha
                        self.state[p]['D'] = D

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
        # Gather stochastic grads
        loss = closure()
        # Reset params and update grads
        for group in self.param_groups:
            gradnorm_sq = 0.
            dot = 0.
            for p in group['params']:
                if 'prev' in self.state[p]:
                    dot += (p.grad * (self.state[p]['prev'] - p)).sum().item()
                    gradnorm_sq += (p.grad * p.grad).sum().item()
                self.state[p]['prev'] = p.detach().clone()
                # precondition grad and take a step
                if 'D' in self.state[p]:
                    p.grad.mul_(self.state[p]['D']**-1)
                p.sub_(p.grad, alpha=group['lr'])
                p.grad.detach_()
                p.grad.zero_()
            if 4 * group['alpha'] * dot >= gradnorm_sq:
                group['reg_const'] = max(0.25 * group['reg_const'], group['reg_const_min'])
            else:
                group['reg_const'] = 4 * group['reg_const']

        self.global_state['t'] += 1
        return loss


class ScaledSVRG(ScaledSGD):
    def __init__(self, params, period=10**10, SARAH=False, **kwargs):
        self.SARAH = SARAH  # change SVRG to SARAH
        super().__init__(params, **kwargs)
        self.global_state.setdefault('ref_evals', 0)  # num of reference updates
        self.global_state.setdefault('should_ref', True)  # we should update reference next step
        self.global_state.setdefault('ref_period', period)  # period of updating reference (in `t`)

    @torch.no_grad()
    def update_ref(self):
        """
        Update reference params and full grad.
        """
        ref_gradnorm = 0.
        for group in self.param_groups:
            for p in group['params']:
                pstate = self.state[p]
                pstate['ref'] = p.detach().clone()
                if p.grad is None:
                    pstate['full_grad'] = torch.zeros_like(pstate['ref'])
                else:
                    pstate['full_grad'] = p.grad.detach().clone()
                    ref_gradnorm += torch.sum(p.grad**2).item()
        self.global_state['ref_gradnorm'] = ref_gradnorm**0.5
        self.global_state['ref_evals'] += 1
        self.global_state['should_ref'] = False

    @torch.no_grad()
    def step(self, closure):
        t = self.global_state['t']
        warmup = self.global_state['warmup']
        ref_period = self.global_state['ref_period']
        should_ref = self.global_state['should_ref'] or (t % ref_period) == 0

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

        ##### Update reference params and full grad #####
        if should_ref:
            print(f"Updating reference...")
            closure(full_batch=True)
            self.update_ref()

        ##### Update diagonal #####
        closure(create_graph=True)
        self.update_diagonal()

        ##### Take step #####
        gradnorm = 0.
        # Gather stochastic grads on orig params
        loss = closure()
        # Store orig params and grads, and set params to ref params
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['orig'] = p.detach().clone()
                self.state[p]['orig_grad'] = p.grad.detach().clone()
                p.set_(self.state[p]['ref'].data)
        # Gather stochastic grads of loss on ref params
        closure()
        # Reset params and update grads
        for group in self.param_groups:
            gradnorm_sq = 0.
            dot = 0.
            for p in group['params']:
                if 'prev' in self.state[p]:
                    dot += (p.grad * (self.state[p]['prev'] - p)).sum().item()
                    gradnorm_sq += (p.grad * p.grad).sum().item()
                self.state[p]['prev'] = p.detach().clone()
                orig_param = self.state[p]['orig']
                ref_grad = p.grad if p.grad is not None else 0.
                orig_grad = self.state[p]['orig_grad']
                full_grad = self.state[p]['full_grad']
                if orig_grad is None:
                    self.state[p]['orig'] = None
                    continue
                grad = full_grad - ref_grad + orig_grad
                ### Add this line for SVRG -> SARAH ###
                if self.SARAH:
                    self.state[p]['ref'] = orig_param.detach().clone()
                    self.state[p]['full_grad'] = grad.detach().clone()
                gradnorm_sq += torch.sum(grad**2).item()
                # precondition grad and take a step
                if 'D' in self.state[p]:
                    grad.mul_(self.state[p]['D']**-1)
                p.set_(orig_param - group['lr'] * grad)
                p.grad.detach_()
                p.grad.zero_()
                self.state[p]['orig'] = None
                self.state[p]['orig_grad'] = None
            if 4 * group['alpha'] * dot >= gradnorm_sq:
                group['reg_const'] = max(0.25 * group['reg_const'], group['reg_const_min'])
            else:
                group['reg_const'] = 4 * group['reg_const']

            gradnorm += gradnorm_sq
        gradnorm = gradnorm**0.5

        self.global_state['should_ref'] = gradnorm < 0.1 * self.global_state['ref_gradnorm']
        self.global_state['t'] += 1
        return loss


class ScaledSARAH(ScaledSVRG):
    def __init__(self, params, **kwargs):
        kwargs['SARAH'] = True
        super().__init__(params, **kwargs)


class ScaledLSVRG(ScaledSVRG):
    def __init__(self, params, p=0.99, **kwargs):
        if 'ref_period' in kwargs:
            del kwargs['ref_period']
        super().__init__(params, **kwargs)
        self.global_state.setdefault('ref_prob', p)  # prob of updating reference
        self.global_state.setdefault('ref_period', 10**10)  # don't use period

    @torch.no_grad()
    def step(self, closure):
        # Should update reference with prob `ref_prob`
        should_ref = torch.rand(1).item() > self.global_state['ref_prob']
        self.global_state['should_ref'] = self.global_state['should_ref'] or should_ref
        return super().step(closure)

