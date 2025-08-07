import torch
import re
from typing import Tuple,List

def cans_ortho(X:torch.Tensor, s_interval:Tuple[float,float], num_iterations:int, poly_degrees:List[int]) -> torch.Tensor:
    """
    Apply Chebyshev polynomial approximation to orthogonalize a matrix X.
    
    Args:
        X (torch.Tensor): Input tensor to be orthogonalized.
        s_interval (Tuple[float, float]): Interval for Chebyshev approximation.
        num_iterations (int): Number of iterations for the approximation.
        poly_degrees (List[int]): Degrees of the Chebyshev polynomials to use.
        
    Returns:
        torch.Tensor: Orthogonalized tensor.
    """
    # Placeholder for actual implementation
    return X  # Replace with actual orthogonalization logic


def delta_ortho(X:torch.Tensor, s_interval:Tuple[float,float], poly_degrees:List[int], delta:float, eps:float=1e-7) -> torch.Tensor:

    # Placeholder for actual implementation
    return X  # Replace with actual orthogonalization logic


def remez(s_interval:Tuple[float,float],poly_degree:int ):
    """
    Compute the Remez coefficients for Chebyshev polynomial approximation.
    
    Args:
        s_interval (Tuple[float, float]): Interval for Chebyshev approximation.
        poly_degree (int): Degree of the Chebyshev polynomial.
        
    Returns:
        List[float]: Coefficients of the Chebyshev polynomial.
    """
    # Placeholder for actual implementation
    pass
















def newtonschulz(G, steps=5, eps=1e-7):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G / (G.norm() + eps)
    transpose = G.size(0) > G.size(1)
    if transpose:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transpose:
        X = X.T
    return X

def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0]**step)
    buf2c = buf2 / (1 - betas[1]**step)
    return buf1c / (buf2c.sqrt() + eps)


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4: # for the case of conv filters
        update = update.view(len(update), -1)
    update = newtonschulz(update, steps=ns_steps)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update


