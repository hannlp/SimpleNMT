class Noam(object):
    def __init__(self, optimizer, lr_scale, d_model, warmup_steps) -> None:
        self._optimizer = optimizer
        self.lr_scale = lr_scale
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.n_steps = 0
    
    def zero_grad(self):
        self._optimizer.zero_grad()

    def step(self):
        self.n_steps += 1
        lrate = self.d_model ** -0.5 * \
            min(self.n_steps ** -0.5, self.n_steps * self.warmup_steps ** -1.5)
        
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = self.lr_scale * lrate
        
        self._optimizer.step()
    
    @property
    def param_groups(self):
        return self._optimizer.param_groups

    @property
    def state_dict(self):
        return self._optimizer.state_dict
