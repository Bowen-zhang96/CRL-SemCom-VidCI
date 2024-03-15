import torch
from torch.autograd import Function

device='cuda:0'

''' Function for the binarized hadamard product between weights and inputs'''
def where(cond, x1, x2):
    return cond.float() * x1 + (1 - cond.float()) * x2


class BinarizeHadamardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        weight_b = where(weight>=0, 1, 0) # binarize weights
        output = input * weight_b
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        weight_b = where(weight>=0, 1, 0) # binarize weights
        grad_input = grad_weight = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output * weight_b
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output * input

        return grad_input, grad_weight


binarize_hadamard = BinarizeHadamardFunction.apply

class SignIncludingZero(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = input >= 0
        return output.float()

    def backward(ctx, grad_output):
        grad_input = torch.clamp(grad_output, -1, 1)
        return grad_input


class SignExcludingZero(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = input > 0
        return output.float()

    def backward(ctx, grad_output):
        grad_input = torch.clamp(grad_output, -1, 1)
        return grad_input


sign_incl0 = SignIncludingZero.apply
sign_excl0 = SignExcludingZero.apply


def less_equal(a, b):  # a <= b
    return sign_incl0(b - a)


def less_than(a, b):  # a < b
    return sign_excl0(b - a)


def greater_equal(a, b):  # a >= b
    return sign_incl0(a - b)


def greater_than(a, b):  # a > b
    return sign_excl0(a - b)
