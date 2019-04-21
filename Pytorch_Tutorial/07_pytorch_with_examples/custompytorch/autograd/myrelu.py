import torch
from torch.autograd import Function

class MyReLU(Function):
    """
    Implement customautograd Functions by subclassing torch.autograd.Function and implement forward and backward passes which operate on Tensors
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward, we receive a Tensor containing input and return an output Tensor
        ctx is a context object that can be used to stash information for backward computation
        ctx.save_for_backward is to cache arbitrary objects in backward
        """
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In backward, we receive a Tensor containing gradient of loss wrt output, then we need to compute gradient of loss wrt input
        """
        input, = ctx.saved_tensors
        grad_input = grad_output * torch.where(input < 0, torch.tensor(0, dtype=torch.float32), torch.tensor(1, dtype=torch.float32))
        return grad_input