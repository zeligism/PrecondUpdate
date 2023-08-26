
import torch


class SVRG(torch.optim.SGD):
    def __init__(self, params, period=10**10, SARAH=False, **kwargs):
        self.SARAH = SARAH  # change SVRG to SARAH
        super().__init__(params, **kwargs)
        self.global_state.setdefault('ref_evals', 0)  # num of reference updates
        self.global_state.setdefault('should_ref', True)  # we should update reference next step
        self.global_state.setdefault('ref_period', period)  # period of updating reference (in `t`)
        self.global_state.setdefault('t', 0)  # num of step iters

    @property
    def global_state(self):
        # First param holds global state
        p0 = self.param_groups[0]['params'][0]
        return self.state[p0]

    @torch.no_grad()
    def update_ref(self):
        """
        Update reference params and full grad.
        `closure(full_batch=True)` must be called before this.
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
        """
        `closure` should support calculating a variance-reduced (reference) gradient.
        See `self.update_ref` where the reference gradient is the full batch gradient.
        """
        t = self.global_state['t']
        ref_period = self.global_state['ref_period']
        should_ref = self.global_state['should_ref'] or (t % ref_period) == 0

        closure = torch.enable_grad()(closure)  # always enable grad for closure

        ##### Update reference params and full grad #####
        if should_ref:
            self.global_state['last_ref_t'] = t - 1
            print(f"Updating reference...")
            closure(full_batch=True)
            self.update_ref()

        ##### Take step #####
        # First, define a variance-reduction closure that sets p.grad to the variance-reduced grad
        @torch.no_grad()
        def vr_closure():
            # Gather stochastic grads on orig params
            loss = closure()
            # Store orig params and grads, and set params to ref params
            for group in self.param_groups:
                for p in group['params']:
                    pstate = self.state[p]
                    pstate['orig'] = p.detach().clone()
                    pstate['orig_grad'] = p.grad.detach().clone()
                    # set to ref params
                    p.copy_(pstate['ref'])
            # Gather stochastic grads of loss on ref params
            closure()
            gradnorm = 0.
            for group in self.param_groups:
                for p in group['params']:
                    pstate = self.state[p]
                    ref_grad = p.grad if p.grad is not None else 0.
                    orig_grad = pstate['orig_grad']
                    full_grad = pstate['full_grad']
                    if orig_grad is None:
                        continue
                    vr_reduced_grad = full_grad.sub(ref_grad).add(orig_grad)
                    # ### Add this line for SVRG -> SARAH ###
                    # if self.SARAH:
                    #     if vr_reduced_grad.pow(2).sum() > pstate['full_grad'].pow(2).sum():
                    #         beta = 1  # beta=0 --> SARAH,  beta=1 --> SVRG
                    #     else:
                    #         beta = 0
                    #     pstate['ref'].mul_(beta).add_(pstate['orig'].mul(1 - beta))
                    #     pstate['full_grad'].mul_(beta).add_(vr_reduced_grad.mul(1 - beta))
                    # set to back to original params
                    p.copy_(pstate['orig'])
                    p.grad.copy_(vr_reduced_grad)
                    gradnorm += torch.sum(p.grad**2).item()
            gradnorm = gradnorm**0.5

            if self.SARAH:
                for group in self.param_groups:
                    for p in group['params']:
                        pstate = self.state[p]
                        beta = 0.0
                        pstate['ref'].mul_(beta).add_(pstate['orig'].mul(1 - beta))
                        pstate['full_grad'].mul_(beta).add_(p.grad.mul(1 - beta))
                self.global_state['ref_gradnorm'] = gradnorm

            self.global_state['should_ref'] = gradnorm < 0.1 * self.global_state['ref_gradnorm']
            return loss

        # Take SGD step with the variance-reduction closure
        loss = super().step(vr_closure)

        self.global_state['t'] += 1
        return loss


class SARAH(SVRG):
    def __init__(self, params, **kwargs):
        kwargs['SARAH'] = True
        super().__init__(params, **kwargs)


class LSVRG(SVRG):
    def __init__(self, params, p=0.99, **kwargs):
        if 'ref_period' in kwargs:
            del kwargs['ref_period']
        super().__init__(params, **kwargs)
        self.global_state.setdefault('ref_prob', p)  # prob of updating reference
        print(f"p = {p:.5f}")
        self.global_state.setdefault('ref_period', 10**10)  # don't use period

    @torch.no_grad()
    def step(self, closure):
        # Should update reference with prob `ref_prob`
        should_ref = torch.rand(1).item() > self.global_state['ref_prob']
        self.global_state['should_ref'] = self.global_state['should_ref'] or should_ref
        return super().step(closure)

