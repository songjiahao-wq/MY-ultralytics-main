import torch
from torch.distributed import ReduceOp
import contextlib

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
import math
import numpy as np

def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)


class ProportionScheduler:
    def __init__(self, pytorch_lr_scheduler, max_lr, min_lr, max_value, min_value):
        """
        This scheduler outputs a value that evolves proportional to pytorch_lr_scheduler, e.g.
        (value - min_value) / (max_value - min_value) = (lr - min_lr) / (max_lr - min_lr)
        """
        self.t = 0
        self.pytorch_lr_scheduler = pytorch_lr_scheduler
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.max_value = max_value
        self.min_value = min_value

        assert (max_lr > min_lr) or ((max_lr == min_lr) and (
                    max_value == min_value)), "Current scheduler for `value` is scheduled to evolve proportionally to `lr`," \
                                              "e.g. `(lr - min_lr) / (max_lr - min_lr) = (value - min_value) / (max_value - min_value)`. Please check `max_lr >= min_lr` and `max_value >= min_value`;" \
                                              "if `max_lr==min_lr` hence `lr` is constant with step, please set 'max_value == min_value' so 'value' is constant with step."

        assert max_value >= min_value

        self.step()  # take 1 step during initialization to get self._last_lr

    def lr(self):
        return self._last_lr[0]

    def step(self):
        self.t += 1
        if hasattr(self.pytorch_lr_scheduler, "_last_lr"):
            lr = self.pytorch_lr_scheduler._last_lr[0]
        else:
            lr = self.pytorch_lr_scheduler.optimizer.param_groups[0]['lr']

        if self.max_lr > self.min_lr:
            value = self.min_value + (self.max_value - self.min_value) * (lr - self.min_lr) / (
                        self.max_lr - self.min_lr)
        else:
            value = self.max_value

        self._last_lr = [value]
        return value


class SchedulerBase:
    def __init__(self, T_max, max_value, min_value=0.0, init_value=0.0, warmup_steps=0, optimizer=None):
        super(SchedulerBase, self).__init__()
        self.t = 0
        self.min_value = min_value
        self.max_value = max_value
        self.init_value = init_value
        self.warmup_steps = warmup_steps
        self.total_steps = T_max

        # record current value in self._last_lr to match API from torch.optim.lr_scheduler
        self._last_lr = [init_value]

        # If optimizer is not None, will set learning rate to all trainable parameters in optimizer.
        # If optimizer is None, only output the value of lr.
        self.optimizer = optimizer

    def step(self):
        if self.t < self.warmup_steps:
            value = self.init_value + (self.max_value - self.init_value) * self.t / self.warmup_steps
        elif self.t == self.warmup_steps:
            value = self.max_value
        else:
            value = self.step_func()
        self.t += 1

        # apply the lr to optimizer if it's provided
        if self.optimizer is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = value

        self._last_lr = [value]
        return value

    def step_func(self):
        pass

    def lr(self):
        return self._last_lr[0]


class LinearScheduler(SchedulerBase):
    def step_func(self):
        value = self.max_value + (self.min_value - self.max_value) * (self.t - self.warmup_steps) / (
                self.total_steps - self.warmup_steps)
        return value


class CosineScheduler(SchedulerBase):
    def step_func(self):
        phase = (self.t - self.warmup_steps) / (self.total_steps - self.warmup_steps) * math.pi
        value = self.min_value + (self.max_value - self.min_value) * (np.cos(phase) + 1.) / 2.0
        return value


class PolyScheduler(SchedulerBase):
    def __init__(self, poly_order=-0.5, *args, **kwargs):
        super(PolyScheduler, self).__init__(*args, **kwargs)
        self.poly_order = poly_order
        assert poly_order <= 0, "Please check poly_order<=0 so that the scheduler decreases with steps"

    def step_func(self):
        value = self.min_value + (self.max_value - self.min_value) * (self.t - self.warmup_steps) ** self.poly_order
        return value
class GAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, model, grad_rho_scheduler=None,
                 grad_norm_rho_scheduler=None, adaptive=False, perturb_eps=1e-12, args=None,
                 grad_reduce='mean',lr=0.0, **kwargs):
        defaults = dict(adaptive=adaptive, **kwargs)
        super(GAM, self).__init__(params, defaults)
        self.grad_rho_scheduler = grad_rho_scheduler
        self.grad_norm_rho_scheduler = grad_norm_rho_scheduler
        self.perturb_eps = perturb_eps
        self.model = model
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.adaptive = adaptive
        self.args = args
        self.get_grad_reduce(grad_reduce)
        self.lr = lr
        self.update_rho_t()

    def get_grad_reduce(self, grad_reduce: str):
        if grad_reduce.lower() == 'mean':
            if hasattr(ReduceOp, 'AVG'):
                self.grad_reduce = ReduceOp.AVG
                self.manual_average = False
            else:  # PyTorch <= 1.11.0 does not have AVG, need to manually average across processes
                self.grad_reduce = ReduceOp.SUM
                self.manual_average = True
        elif grad_reduce.lower() == 'sum':
            self.grad_reduce = ReduceOp.SUM
            self.manual_average = False
        else:
            raise ValueError('"grad_reduce" should be one of ["mean", "sum"].')

    @torch.no_grad()
    def update_rho_t(self):
        if self.grad_rho_scheduler is not None:
            self.grad_rho = self.grad_rho_scheduler.step()
        if self.grad_norm_rho_scheduler is not None:
            self.grad_norm_rho = self.grad_norm_rho_scheduler.step()

    @torch.no_grad()
    def perturb_weights(self, perturb_idx: int):
        grad_norm = self._grad_norm(weight_adaptive=self.adaptive)
        scale = self.grad_rho / (grad_norm + self.perturb_eps)

        if perturb_idx == 0:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    self.state[p]["g_0"] = p.grad.data.clone()
                    e_w = p.grad * scale.to(p)
                    if self.adaptive:
                        e_w *= torch.pow(p, 2)
                    p.add_(e_w)
                    self.state[p]['e_w_0'] = e_w

        elif perturb_idx == 1:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    self.state[p]["g_2"] = p.grad.data.clone()
                    e_w = p.grad * scale.to(p)
                    if self.adaptive:
                        e_w *= torch.pow(p, 2)
                    p.add_(e_w)
                    self.state[p]['e_w_1_2'] += e_w

        else:
            raise ValueError('"perturb_idx" should be one of [0, 1].')

    @torch.no_grad()
    def grad_norm_ascent(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["g_1"] = p.grad.data.clone()
                p.grad.data -= self.state[p]["g_0"]

        grad_norm = self._grad_norm(weight_adaptive=self.adaptive)
        scale = self.grad_norm_rho / (grad_norm + self.perturb_eps)

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                if self.adaptive:
                    e_w *= torch.pow(p, 2)
                p.add_(e_w)
                self.state[p]['e_w_1_2'] = e_w

    @torch.no_grad()
    def unperturb(self, perturb_key: str):

        for group in self.param_groups:
            for p in group['params']:
                if perturb_key in self.state[p].keys():
                    p.data.sub_(self.state[p][perturb_key])

    @torch.no_grad()
    def gradient_decompose(self, args=None):
        inner_prod = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                # update the weighted sum of grads
                self.state[p]['pro_m'] = self.state[p]['g_0'] + abs(args.grad_beta_2) * self.state[p]['g_2']
                p.grad.data = args.grad_beta_1 * self.state[p][
                    "g_1"] + args.grad_beta_3 * p.grad.data.detach().clone()
                inner_prod += torch.sum(
                    self.state[p]['pro_m'] * p.grad.data
                )

        # get norm
        new_grad_norm = self._grad_norm()
        old_grad_norm = self._grad_norm(by='pro_m')

        # get cosine
        cosine = inner_prod / (new_grad_norm * old_grad_norm + self.perturb_eps)

        # gradient decomposition
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                vertical = self.state[p]['pro_m'] - cosine * old_grad_norm * p.grad.data / (
                        new_grad_norm + self.perturb_eps)
                p.grad.data.add_(vertical, alpha=-args.grad_gamma)

    @torch.no_grad()
    def _grad_norm(self, weight_adaptive: bool = False, by: str = 'grad'):
        norm = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                if by == 'grad':
                    g = p.grad.data
                elif by == 'pro_m':
                    g = self.state[p]['pro_m']
                # elif by == 'e_w':
                #     g = self.state[p]['e_w_0'] + self.state[p]['e_w_1_2'] + self.state[p]['e_w_2']
                elif by == 'p':
                    g = p.data
                else:
                    raise ValueError("Invalid 'by' argument in _grad_norm")

                if weight_adaptive:
                    norm += torch.sum((g * torch.abs(p.data)) ** 2)
                else:
                    norm += torch.sum(g ** 2)

        return torch.sqrt(norm)

    @torch.no_grad()
    def _sync_grad(self):
        if torch.distributed.is_initialized():  # synchronize final gardients
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    if self.manual_average:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
                        world_size = torch.distributed.get_world_size()
                        p.grad.div_(float(world_size))
                    else:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
        return

    def maybe_no_sync(self):
        if torch.distributed.is_initialized():
            return self.model.no_sync()
        else:
            return contextlib.ExitStack()

    @torch.no_grad()
    def set_closure(self, loss_fn, inputs, targets, **kwargs):
        # create self.forward_backward_func, which is a function such that
        # self.forward_backward_func() automatically performs forward and backward passes.

        def get_grad():
            self.zero_grad()
            with torch.enable_grad():
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets, **kwargs)
            loss_value = loss.data.clone().detach()
            loss.backward()
            return outputs, loss_value

        self.forward_backward_func = get_grad

    def step(self, closure=None):

        if closure:
            get_grad = closure
        else:
            get_grad = self.forward_backward_func

        with self.maybe_no_sync():
            # get gradient
            outputs, loss_value = get_grad()

            # perturb weights
            self.perturb_weights(perturb_idx=0)

            # disable running stats for second pass,
            disable_running_stats(self.model)
            # model 1
            get_grad()
            # grad 1

            self.unperturb(perturb_key="e_w_0")
            # model 0
            self.grad_norm_ascent()
            # model 2
            get_grad()
            # grad 2

            self.perturb_weights(perturb_idx=1)
            # model 3
            get_grad()
            # grad 3

            # decompose and get new update direction
            self.gradient_decompose(args=self.args)

            # unperturb
            self.unperturb(perturb_key="e_w_1_2")

        # synchronize gradients across workers
        self._sync_grad()

        # update with new directions
        self.base_optimizer.step()

        # enable running stats
        enable_running_stats(self.model)

        return outputs, loss_value

    def zero_grad(self, set_to_none: bool = False):
        self.base_optimizer.zero_grad(set_to_none)

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)

    # def add_param_group(self, param_group):
    #     self.base_optimizer.add_param_group(param_group)

    def __repr__(self):
        return f'GAM({self.base_optimizer.__class__.__name__})'