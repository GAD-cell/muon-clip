import torch
import re
from typing import Tuple,List


def gelfand_upper_bound(X:torch.Tensor, k:int=2, eps:float=1e-7) -> (torch.tensor, torch.Tensor):
    norm = torch.norm(X,dim=(-1, -2)) + eps
    Y = X / norm   
    A = Y @ Y.T
    Ak = torch.linalg.matrix_power(A, k)
    frob_norm = torch.norm(Ak, dim=(-1, -2))
    upper_bound = frob_norm**(1/(2*k)) + eps # spectral radius < frob_norm^(1/(2*k))
    if torch.isnan(upper_bound):
        print(f"Warning: NaN upper bound for X with shape {X.shape}, norm={norm}, frob_norm={frob_norm}")
        return Y , A 
    return Y/upper_bound, A


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


    for i in range(len(poly_degrees)):
        if poly_degrees[i] == 3:
            poly, error = order_three_poly(s_interval) # use special formula for degree 3
            s_interval = (1-error, 1+error)
            X = poly[1] * X + poly[3] * X @ X.T @ X
        else :
            poly, error = remez(s_interval, poly_degrees[i])
            s_interval = (1-error, 1+error)
            X = evaluate_polynomial(poly, X)

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
            poly, error = remez((a,b), poly_degrees[i])
            a,b = (1-error), (1+error)
            polynomials.append(poly)

        if error < delta:
            r = (l+r)/2
        else:
            l = (l+r)/2
    a,b = 1-delta, 1+delta
    X = compose_polynomials_optimized(polynomials, X)
    return X  

def order_three_poly(s_interval: Tuple[float,float]) -> Tuple[List[float], float]:
    """
    Compute the coefficients for a third-order polynomial approximation.
    
    poly = alpha*x^3 + beta*x^2 + gamma*x + delta
    
    Args:
        s_interval (Tuple[float, float]): Interval for Chebyshev approximation.
        
    Returns:
        Tuple[List[float], float]: Coefficients of the polynomial and the error.
    """
    A, B = s_interval
    
    # Explicit formula for optimal 3rd order polynomial on segment [A, B]
    e = ((A**2 + A * B + B**2) / 3)**(1/2)
    a = 2 / (2 * e**3 + A**2 * B + B**2 * A)
    
    # Polynomial coefficients: -a*x^3 + 0*x^2 + a*(A^2 + A*B + B^2)*x + 0
    delta = 0.0  
    gamma = a * (A**2 + A * B + B**2) 
    beta = 0.0     
    alpha = -a  
    
    error = (2 * e**3 - A**2 * B - B**2 * A) / (2 * e**3 + A**2 * B + B**2 * A)
    
    poly = [delta, gamma, beta, alpha]
    return poly, error

def remez(s_interval:Tuple[float,float], poly_degree:int):
    """
    Compute the Remez coefficients for Chebyshev polynomial approximation. Approximates the constant function 1 on the interval [s_interval[0], s_interval[1]].
    
    Args:
        s_interval (Tuple[float, float]): Interval for Chebyshev approximation.
        poly_degree (int): Degree of the Chebyshev polynomial.
        
    Returns:
        List[float]: Coefficients of the Chebyshev polynomial.
    """
    # Placeholder for actual implementation
    pass


def tensor_power(X, degree):
    """
    Compute tensor power: X^degree using tensor products.
    Alternates between X and X^T: X ⊗ X^T ⊗ X ⊗ X^T ⊗ ...
    """
    if degree == 0:
        return torch.tensor(1.0, device=X.device, dtype=X.dtype)
    elif degree == 1:
        return X
    
    result = X
    for d in range(2, degree + 1):
        if d % 2 == 0:
            result = torch.tensordot(result, X.T, dims=0)
        else:
            result = torch.tensordot(result, X, dims=0)
    
    return result

def evaluate_polynomial(coefficients, X):
    """
    Evaluate polynomial p(X) = a0*I + a1*X + a2*X^2 + ... + an*X^n.
    
    Args:
        coefficients: List of coefficients [a0, a1, ..., an]
        X: Square matrix
    
    Returns:
        Result matrix p(X)
    """
    result = torch.zeros_like(X, device=X.device, dtype=X.dtype)
    I = torch.eye(X.shape[0], device=X.device, dtype=X.dtype)
    
    for i, coeff in enumerate(coefficients):
        if i == 0:
            result += coeff * I
        else:
            result += coeff * tensor_power(X, i)
    
    return result

def newton_schulz_accelerated(G:torch.Tensor, poly_degree:List[float]=None, order:int=3, estimate_lower_bound:int=False, lower_bound=1e-3, iter:int=9) -> torch.Tensor:

    G, A = gelfand_upper_bound(G)
    if estimate_lower_bound: s_interval = (None, 1)
    else: s_interval = (lower_bound, 1)

    if poly_degree is None:
        poly_degree = [order] * iter
    
    X = cans_ortho(G, s_interval, poly_degree)
    return X

def newtonschulz(G, steps=5, eps=1e-7):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G / (G.norm() + eps)
    
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    return X

def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0]**step)
    buf2c = buf2 / (1 - betas[1]**step)
    return buf1c / (buf2c.sqrt() + eps)


def muon_update(grad, momentum, beta:float=0.95, ns_steps:int=9, nesterov:bool=True, better_newton:bool=False):
    if grad.ndim != 2:
        raise ValueError(f"Input tensor must be 2D but got shape {X.shape}")
    
    momentum.lerp_(grad, 1 - beta)
    grad = grad.lerp_(momentum, beta) if nesterov else momentum
    if grad.ndim == 4: # for the case of conv filters
        grad = grad.view(len(update), -1)

    is_transpose = grad.size(0) > grad.size(1)
    if is_transpose: grad = grad.T  
    if better_newton: grad = newton_schulz_accelerated(grad, iter=ns_steps)
    else: grad = newtonschulz(grad, steps=ns_steps)
    grad *= max(1, grad.size(-2) / grad.size(-1))**0.5
    if is_transpose: grad = grad.T

    return grad






