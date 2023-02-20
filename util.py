import torch

SQRT_CONST = 1e-10

def torch_safe_sqrt(x, sqrt_const=SQRT_CONST):
    return torch.sqrt(torch.clamp(x, min=sqrt_const))

def torch_pdist2(X, Y):
    nx = torch.sum(torch.square(X), dim=1, keepdim=True)
    ny = torch.sum(torch.square(Y), dim=1, keepdim=True)
    C = -2 * torch.matmul(X, torch.t(Y))
    D = (C + torch.t(ny)) + nx
    return D
    
