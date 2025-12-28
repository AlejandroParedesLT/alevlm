from torch import nn
import torch
from torch import optim
import typing
from einops import rearrange, einsum
import math
from typing import List, Iterable

class SGD(optim.Optimizer):
    def __init__(
            self, 
            params,
            lr,
            *args,
            **qwargs
        ):
        self.params=params
        if lr<0:
            raise ValueError('Invalid learning rate: {lr}')
        defaults={'lr':lr}
        super().__init__(params,defaults)
        
    def step(self,closure):
        loss=None if closure is None else closure()
        for group in self.param_groups:
            lr=group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                state=self.state[p]
                t=state.get('t',0)
                grad=p.grad.data
                p.data-=lr/math.sqrt(t+1)*grad
                state['t']=t+1
        return loss           
    
class AdamW(optim.Optimizer):
    def __init__(
        self,
        params,
        lr:float=1e-3,
        weight_decay:float=0.1,
        betas:float=(0.9,0.999),
        eps:float=10e-8
    ) -> None:
        if isinstance(lr, torch.Tensor):
            raise ValueError('Learning Rate as tensor is not supported')
        defaults={
            'lr':lr,
            'betas':betas,
            'epsilon':eps,
            'weight_decay':weight_decay
        }
        super().__init__(params,defaults)
    
    def step(self,closure=None):
        """
        Performs a single optimization step
        Args: Closure (Callable optional)
        """
        loss=None if closure is None else closure()
        # for t in range(self.T):
        for group in self.param_groups:
            lr=group['lr']
            beta1,beta2=group['betas']
            eps=group['epsilon']
            wd=group['weight_decay']
            for p in group['params']: # For each layer
                if p.grad is None:
                    continue
                grad=p.grad.data
                state=self.state[p]
                if len(state)==0:
                    state['t']=0
                    state['m']=torch.zeros_like(p.data)
                    state['v']=torch.zeros_like(p.data)
                m=state['m']
                v=state['v']
                t = state.get('t', 0) + 1
                m.mul_(beta1).add_(grad, alpha=1-beta1)
                v.mul_(beta2).addcmul_(grad,grad,value=1-beta2)
                m_hat=m/(1-beta1**t)
                v_hat=v/(1-beta2**t)
                
                p.data.add_(-lr*wd*p.data)
                p.data.add_(m_hat/(v_hat.sqrt()+eps),alpha=-lr)
                state['t']=t
        return loss
                    


class CosineAnnealingWarmup:
    def __init__(
        self,
        optimizer,
        warmup_iters: int,
        max_iters: int,
        min_lr: float,
        max_lr: float,
    ):
        """
        Cosine annealing learning rate scheduler with warmup.
        
        Args:
            optimizer: PyTorch optimizer
            warmup_iters: Number of warmup iterations (T_w)
            max_iters: Total iterations for training (T_c)
            min_lr: Minimum learning rate (alpha_min)
            max_lr: Maximum learning rate after warmup (alpha_max)
        """
        self.optimizer = optimizer
        self.warmup_iters = warmup_iters  # T_w
        self.max_iters = max_iters        # T_c
        self.min_lr = min_lr              # alpha_min
        self.max_lr = max_lr              # alpha_max
        self.current_iter = 0
    
    def step(self):
        """Update learning rate and increment iteration counter."""
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.current_iter += 1
    
    def get_lr(self):
        """
        Calculate current learning rate based on iteration.
        Implements the formula from your paper.
        """
        t = self.current_iter
        
        # Phase 1: Linear warmup (t < T_w)
        if t < self.warmup_iters:
            return (t / self.warmup_iters) * self.max_lr
        
        # Phase 2: Cosine annealing (T_w <= t <= T_c)
        elif t <= self.max_iters:
            progress = (t - self.warmup_iters) / (self.max_iters - self.warmup_iters)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return self.min_lr + cosine_decay * (self.max_lr - self.min_lr)
        
        # Phase 3: After training ends (t > T_c)
        else:
            return self.min_lr
    
    def get_last_lr(self):
        """Return the last computed learning rate (for logging)."""
        return self.get_lr()


def gradient_cliping(parameters: List[torch.Tensor], max_l2_norm: float, M: float, eps: float = 1e-6):
    for param in parameters:
        if param.grad is not None:
            if torch.norm(param.grad) > max_l2_norm:
                param.grad.data.mul_(M / (torch.norm(param.grad) + eps))
