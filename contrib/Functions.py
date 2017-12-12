# MIT License
# 
# Copyright (c) 2017 Max W. Y. Lam
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math

import torch
from torch.autograd import Function


class SGPAFunction(Function):

    @staticmethod
    def forward(ctx, input, alpha_mean, alpha_lgstd, hyper_norm=None):
        ctx.save_for_backward(input, alpha_mean, alpha_lgstd, hyper_norm)
        alpha_std = torch.exp(alpha_lgstd)
        if hyper_norm is None:
            epsilon = torch.randn(input.size(1)*2, alpha_lgstd.size(1))
        else:
            epsilon = torch.rand(input.size(1)*2, alpha_lgstd.size(1))
            l1, l2 = math.tanh(hyper_norm[0])*.5+.5, math.exp(hyper_norm[1])
            epsilon = torch.sinh(epsilon*l2)/torch.cosh(epsilon*l2)**l1/l2
        A1, A2 = (alpha_mean+epsilon*alpha_lgstd).chunk(2)
        Z1, Z2 = input.mm(A1), input.mm(A2)
        S1, S2 = torch.cos(Z1)+torch.cos(Z2), torch.sin(Z1)+torch.sin(Z2)
        output = torch.cat([S1, S2], 1)/math.sqrt(alpha_std.size(0))
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        input, alpha_mean, alpha_lgstd, hyper_norm = ctx.saved_variables
        grad_input = grad_alpha_mean = grad_alpha_lgstd = grad_hyper_norm = None
        
        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        # if ctx.needs_input_grad[0]:
        #     grad_input = grad_output.mm(weight)
        # if ctx.needs_input_grad[1]:
        #     grad_alpha_mean = grad_output.t().mm(input)
        # if bias is not None and ctx.needs_input_grad[2]:
        #     grad_alpha_lgstd = grad_output.sum(0).squeeze(0)
        
        return grad_input, grad_alpha_mean, grad_alpha_lgstd, grad_hyper_norm