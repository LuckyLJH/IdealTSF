import torch
import torch.nn as nn


class RevIN(nn.Module):
    """
    Conditional Reversible Instance Normalization (Conditional RevIN)
    This version of RevIN introduces a conditional input to modulate the normalization parameters.
    """

    def __init__(self, num_features: int, condition_dim: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param condition_dim: the dimension of the condition vector (e.g., class embedding)
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.condition_dim = condition_dim
        self.eps = eps
        self.affine = affine

        if self.affine:
            self._init_params()

        # Condition-based linear transformation for mean and stdev
        self.condition_to_mean = nn.Linear(self.condition_dim, self.num_features)
        self.condition_to_stdev = nn.Linear(self.condition_dim, self.num_features)

    def forward(self, x, mode: str, condition=None):
        if mode == 'norm':
            self._get_statistics(x, condition)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x, condition)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # Initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x, condition):
        # Use condition to generate mean and stdev
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

        # Modulate mean and stdev based on the condition
        if condition is not None:
            condition_mean = self.condition_to_mean(condition).unsqueeze(-1).unsqueeze(-1)
            condition_stdev = self.condition_to_stdev(condition).unsqueeze(-1).unsqueeze(-1)
            self.mean = self.mean + condition_mean
            self.stdev = self.stdev + condition_stdev

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x, condition):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x
