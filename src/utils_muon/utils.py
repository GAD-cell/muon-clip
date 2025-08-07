import torch
import re
from typing import Tuple,List


def gelfand_upper_bound(X:torch.Tensor, k:int=3) -> float, torch.Tensor:
    if X.ndim != 2:
        raise ValueError(f"Input tensor must be 2D., got shape {X.shape}")

    A = X @ X.T
    Ak = torch.linalg.matrix_power(A, k)
    frob_norm = torch.linalg.norm(Ak, ord='fro')
    upper_bound = frob_norm**(1/(2*k))
    return upper_bound, A # return A to avoid overhead of recomputing it later

def cans_ortho(X:torch.Tensor, s_interval:Tuple[float,float], poly_degrees:List[int]) -> torch.Tensor:
    """
    Apply Chebyshev polynomial approximation to orthogonalize a matrix X.
    
    Args:
        X (torch.Tensor): Input tensor to be orthogonalized,souhld be normalized.
        s_interval (Tuple[float, float]): Interval for Chebyshev approximation.
        poly_degrees (List[int]): Degrees of the Chebyshev polynomials to use, must be odd, at least 3.
        
    Returns:
        torch.Tensor: Orthogonalized tensor.
    """
    a = s_interval[0]
    if not a:
        X, a, b = delta_ortho(X, s_interval[1], poly_degrees)
        s_interval = (a, b)
    
    polynomial = []
    for i in range(len(poly_degrees)):
        if poly_degrees[i] == 3:
            
            pass # use special formula for degree 2
        else :
            poly, error = remez(s_interval, poly_degrees[i])
            s_interval = (1-error, 1+error)
            polynomial.append(poly)

    return X  


def delta_ortho(X:torch.Tensor, poly_degrees:List[int], b:float=1.99, max_step:int=10, delta:float=0.99, eps:float=1e-7) -> torch.Tensor:
    """

    """

    l,r = 0, b
    error = 1
    while abs(delta-error) > eps: #do first round,if doesn't fall in the range, then starts again with new boundaries that we know are too conservative
        a,b = (l+r)/2,b
        polynomials = []
        for i in range(len(poly_degrees)):
            poly, error = remez((a,b), 2*poly_degrees[i]-1)
            a,b = (1-error), (1+error)
            polynomials.append(poly)

        if error < delta:
            r = (l+r)/2
        else:
            l = (l+r)/2
    a,b = 1-delta, 1+delta
    X = poly_mul(polynomials, X)
    return X  


def remez(s_interval:Tuple[float,float], poly_degree:int):
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


def poly_mul(polynomials:List[List[float]], X:torch.Tensor) -> torch.Tensor:
    """
    Placeholder for polynomial multiplication function.
    
    Returns:
        torch.Tensor: Result of polynomial multiplication.
    """
    # Placeholder for actual implementation
    return X


def newton_schulz_accelerated(G:torch.Tensor, poly_degree:List[float]=None, order:int=3, estimate_lower_bound:int=False, lower_bound=1e-3, num_iterations:int=9) -> torch.Tensor:
    upper_bound, A = gelfand_upper_bound(G)
    if estimate_lower_bound: s_interval = (None, upper_bound)
    else: s_interval = (lower_bound, upper_bound)

    if poly_degree is None:
        poly_degree = [order] * num_iterations
    
    X = cans_ortho(G, s_interval, poly_degree)



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




