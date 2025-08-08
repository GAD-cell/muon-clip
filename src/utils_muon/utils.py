import torch
import re
from typing import Tuple,List


def gelfand_upper_bound(X:torch.Tensor, k:int=2) -> (torch.tensor, torch.Tensor):
    A = X @ X.T
    Ak = torch.linalg.matrix_power(A, k)
    frob_norm = torch.linalg.norm(Ak, ord='fro')
    upper_bound = frob_norm**(1/(2*k)) # spectral radius < frob_norm^(1/(2*k))
    normalize_X = X / upper_bound # normalize X to have spectral radius <= 1
    return normalize_X, A # return A to avoid overhead of recomputing it later

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

    polynomials = []
    for i in range(len(poly_degrees)):
        if poly_degrees[i] == 3:
            poly, error = order_three_poly(s_interval) # use special formula for degree 3
            s_interval = (1-error, 1+error)
            polynomials.append(poly)
        else :
            poly, error = remez(s_interval, poly_degrees[i])
            s_interval = (1-error, 1+error)
            polynomials.append(poly)

    X = compose_polynomials_optimized(polynomials, X) # compose the polynomials
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
    X = compose_polynomials_optimized(polynomials, X)
    return X  

def order_three_poly(s_interval:Tuple[float,float]) -> Tuple[List[float], float]:
    """
    Compute the coefficients for a third-order polynomial approximation.
    
    poly = alpha*x^3 + beta*x^2 + gamma*x + delta
    
    Args:
        s_interval (Tuple[float, float]): Interval for Chebyshev approximation.
        
    Returns:
        Tuple[List[float], float]: Coefficients of the polynomial and the error.
    """
    a, b = s_interval

    beta = 0.0
    delta = 0.0

    fac = 2 * ((a**2 + a *b + b**2)/3)**(3/2)
    h = 2 / (fac + a**2 * b + a * b**2)
    alpha = -h
    gamma = h * (a**2 + a * b + b**2)

    error = (fac - a**2 * b - a * b**2) / (fac + a**2 * b + a * b**2)
    poly = [delta, gamma, beta, alpha]
    return poly, error

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

def evaluate_cubic_polynomial(coefficients, X):
    """
    Evaluate cubic polynomial p(X) = a0*I + a1*X + a2*X^2 + a3*X^3.
    Optimized for degree 3 polynomials.
    
    Args:
        coefficients: List of 4 coefficients [a0, a1, a2, a3]
        X: Square matrix
    
    Returns:
        Result matrix p(X)
    """
    a0, a1, a2, a3 = coefficients
    
    I = torch.eye(X.shape[0], device=X.device, dtype=X.dtype)
    X2 = X @ X
    X3 = X2 @ X
    
    return a0 * I + a1 * X + a2 * X2 + a3 * X3

def compose_polynomials_optimized(polynomial_coefficients, input_tensor):
    """
    Apply composition of degree-3 matrix polynomials.
    
    Args:
        polynomial_coefficients: List of lists, each with 4 coefficients [a0, a1, a2, a3]
        input_tensor: PyTorch tensor of shape (n, m)
    
    Returns:
        Result matrix after applying the polynomial composition
    """
    if not polynomial_coefficients:
        return input_tensor
    
    current_values = input_tensor.clone()
    
    for coefficients in polynomial_coefficients:
        if not coefficients or len(coefficients) != 4:
            continue
        
        # For Newton-Schulz orthogonalization with rectangular matrices
        if current_values.shape[0] != current_values.shape[1]:
            A = current_values @ current_values.T
            A_result = evaluate_cubic_polynomial(coefficients, A)
            current_values = A_result @ current_values
        else:
            # For square matrices
            current_values = evaluate_cubic_polynomial(coefficients, current_values)
    
    return current_values


def newton_schulz_accelerated(G:torch.Tensor, poly_degree:List[float]=None, order:int=3, estimate_lower_bound:int=False, lower_bound=1e-3, iter:int=9) -> torch.Tensor:
    if G.ndim != 2:
        raise ValueError(f"Input tensor must be 2D but got shape {X.shape}")

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


def muon_update(grad, momentum, beta:float=0.95, ns_steps:int=5, nesterov:bool=True, better_newton:bool=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4: # for the case of conv filters
        update = update.view(len(update), -1)
    
    is_transpose = update.size(0) > update.size(1)
    if is_transpose:
        update = update.T  

    if better_newton: update = newton_schulz_accelerated(update, iter=ns_steps)
    else: update = newtonschulz(update, steps=ns_steps)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    
    if is_transpose:
        update = update.T
    return update






