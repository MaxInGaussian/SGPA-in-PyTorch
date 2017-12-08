
import math
import torch as to
import torch.nn.functional as F
from torch.nn import Parameter, Module

from Functions import SGPAFunction

class SGPA(Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = Ax + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, in\_features)` where `*` means any number of
          additional dimensions
        - Output: :math:`(N, *, out\_features)` where all but the last dimension
          are the same shape as the input.

    Attributes:
        weight: the learnable weights of the module of shape
            (out_features x in_features)
        bias:   the learnable bias of the module of shape (out_features)

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = autograd.Variable(torch.randn(128, 20))
        >>> output = m(input)
        >>> print(output.size())
    """

    def __init__(self, in_features, out_features, bias=True):
        super(SGPA, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha_mean = Parameter(to.Tensor(2*in_features, out_features//2))
        self.alpha_logstd = Parameter(to.Tensor(2*in_features, out_features//2))
        self.quasi_norm = Parameter(to.Tensor(2))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1./math.sqrt(self.alpha_mean.size(1))
        self.alpha_mean.data.uniform_(-stdv, stdv)
        self.alpha_logstd.data.uniform_(-stdv, stdv)
        self.quasi_norm.data.normal_()

    def forward(self, input):
        return SGPAFunction.apply(
            input, self.alpha_mean, self.alpha_logstd, self.quasi_norm)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) + ')'