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
import torch as to
from torch.autograd import Variable, Function
from torch.nn import Parameter, Module


class SGPA(Module):
    
    r"""Applies a linear transformation to the incoming data: :math:`y = Ax + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        hyper_normal: If set to False, the layer will use randn instead of rand.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, in\_features)` where `*` means any number of
          additional dimensions
        - Output: :math:`(N, *, out\_features)` where all but the last dimension
          are the same shape as the input.

    Attributes:
        alpha_mean: the learnable weights of the module of shape
            (2*in_features x out_features//2)
        alpha_lgstd: the learnable weights of the module of shape
            (2*in_features x out_features//2)
        hyper_norm: two learnable parameters for random normal sampling

    Examples::

        >>> m = SGPA(20, 30)
        >>> input = autograd.Variable(torch.randn(128, 20))
        >>> output = m(input)
        >>> print(output.size())
    """

    def __init__(self, in_features, out_features, hyper_normal=True):
        super(SGPA, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha_mean = Parameter(to.Tensor(2*in_features, out_features//2))
        self.alpha_lgstd = Parameter(to.Tensor(2*in_features, out_features//2))
        if hyper_normal:
            self.hyper_norm = Parameter(to.Tensor(2))
        else:
            self.register_parameter('hyper_norm', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1./math.sqrt(self.alpha_mean.size(1))
        self.alpha_mean.data.uniform_(-stdv, stdv)
        self.alpha_lgstd.data.uniform_(-stdv, stdv)
        if self.hyper_norm is not None:
            self.hyper_norm.data.normal_()

    def forward(self, input):
        return SGPAFunction.apply(
            input, self.alpha_mean, self.alpha_lgstd, self.hyper_norm)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) + ')'


class SGPAFunction(Function):

    @staticmethod
    def forward(ctx, input, alpha_mean, alpha_lgstd, hyper_norm=None):
        ctx.save_for_backward(input, alpha_mean, alpha_lgstd, hyper_norm)
        alpha_std = to.exp(alpha_lgstd)
        if hyper_norm is None:
            epsilon = to.randn(*alpha_std.size())
        else:
            epsilon = to.rand(*alpha_std.size())
            l1, l2 = math.tanh(hyper_norm[0])*.5+.5, math.exp(hyper_norm[1])
            epsilon = to.sinh(epsilon*l2)/to.cosh(epsilon*l2)**l1/l2
        A1, A2 = (alpha_mean+epsilon*alpha_lgstd).chunk(2)
        Z1, Z2 = input.mm(A1), input.mm(A2)
        S1, S2 = to.cos(Z1)+to.cos(Z2), to.sin(Z1)+to.sin(Z2)
        output = to.cat([S1, S2], 1)/math.sqrt(alpha_std.size(0))
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