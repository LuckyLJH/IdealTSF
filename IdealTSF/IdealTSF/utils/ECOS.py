import torch
from torch.optim import Optimizer

class ECOS(Optimizer):


    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, num_steps=3, epsilon=0.1, alpha=0.5, incremental=False, low_order=False, **kwargs):

        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, num_steps=num_steps, epsilon=epsilon, alpha=alpha, incremental=incremental, low_order=low_order, **kwargs)
        super(ECOS, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):

        grad_norm = self._grad_norm()

        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue


                if group["low_order"]:
                    grad = p.grad.sign()
                else:
                    grad = p.grad

                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * grad * scale.to(p)
                p.add_(e_w)  # "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])  #  "w"

        self.base_optimizer.step()  # "sharpness-aware"

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, model, x, y, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)

        x_adv = self.adversarial_step(model, x, y, epsilon=0.1, alpha=0.05, num_iter=3)

        self.first_step(zero_grad=True)

        for _ in range(self.param_groups[0]["num_steps"]):
            closure()
            self.base_optimizer.step()

        self.second_step()

        model.zero_grad()
        output = model(x_adv)
        loss = torch.nn.functional.mse_loss(output, y)
        loss.backward()

        self.base_optimizer.step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack(
                [
                    ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad)
                    .norm(p=2)
                    .to(shared_device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2,
        )
        return norm

    def adversarial_step(self, model, x, y, epsilon, alpha, num_iter=3):

        x_adv = x.clone().detach().requires_grad_(True).to(x.device)
        for _ in range(num_iter):
            output = model(x_adv)
            loss = torch.nn.functional.mse_loss(output, y)

            model.zero_grad()
            loss.backward()

            grad = x_adv.grad
            x_adv = x_adv + alpha * grad.sign()

            x_adv = torch.clamp(x_adv, x - epsilon, x + epsilon)
            x_adv = torch.clamp(x_adv, 0, 1)
            x_adv = x_adv.detach().requires_grad_(True)

        return x_adv
