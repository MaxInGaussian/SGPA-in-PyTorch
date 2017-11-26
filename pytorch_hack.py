import torch

def matmul(A, B):
    return torch.bmm(A, B.unsqueeze(0).expand(A.size(0), *B.size()))
