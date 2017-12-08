import math
import torch as to
from torch.autograd import Variable


class SGPAFunction(to.autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input, alpha_mean, alpha_logstd, quasi_norm):
        ctx.save_for_backward(input, alpha_mean, alpha_logstd, quasi_norm)
        alpha_std = to.exp(alpha_logstd)
        # epsilon = to.randn(*alpha_std.size())
        epsilon = to.rand(*alpha_std.size())
        l1, l2 = math.tanh(quasi_norm[0])*.5+.5, math.exp(quasi_norm[1])
        epsilon = to.sinh(epsilon*l2)/to.cosh(epsilon*l2)**l1/l2
        A1, A2 = (alpha_mean+epsilon*alpha_logstd).chunk(2)
        Z1, Z2 = input.mm(A1), input.mm(A2)
        S1, S2 = to.cos(Z1)+to.cos(Z2), to.sin(Z1)+to.sin(Z2)
        output = to.cat([S1, S2], 1)/math.sqrt(alpha_std.size(0))
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, alpha_mean, alpha_logstd, quasi_norm = ctx.saved_variables
        grad_input = grad_alpha_mean = grad_alpha_logstd = grad_quasi_norm = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        # if ctx.needs_input_grad[0]:
        #     grad_input = grad_output.mm(weight)
        # if ctx.needs_input_grad[1]:
        #     grad_alpha_mean = grad_output.t().mm(input)
        # if bias is not None and ctx.needs_input_grad[2]:
        #     grad_alpha_logstd = grad_output.sum(0).squeeze(0)

        return grad_input, grad_alpha_mean, grad_alpha_logstd, grad_quasi_norm