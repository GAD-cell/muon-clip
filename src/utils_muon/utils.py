import torch
import re
from typing import Tuple,List

class OrthoPolynomials:
    init_polynomials = []
    init_interval = (None, None)

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


def cans_ortho(X:torch.Tensor, s_interval:Tuple[float,float],ortho_polynomials, poly_degrees:List[int]) -> torch.Tensor:
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
    if not a and not ortho_polynomials.init_polynomials:
        ortho_polynomials.init_polynomials, a, b = delta_ortho(poly_degrees=poly_degrees, b=s_interval[1])
        #print(ortho_polynomials.init_polynomials)
        ortho_polynomials.init_interval = (a, b)

    if ortho_polynomials.init_polynomials:
        for i in range(len(ortho_polynomials.init_polynomials)):
            X = ortho_polynomials.init_polynomials[i][2] * X + ortho_polynomials.init_polynomials[i][0] * X @ X.T @ X
    
    s_interval = ortho_polynomials.init_interval if ortho_polynomials.init_interval else s_interval
    for i in range(len(poly_degrees)):
        if poly_degrees[i] == 3:
            poly, error = order_three_poly(s_interval) # use special formula for degree 3
            #print(poly, error)
            s_interval = (1-error, 1+error)
            X = poly[2] * X + poly[0] * X @ X.T @ X
        else :
            poly, error = remez(s_interval, poly_degrees[i])
            s_interval = (1-error, 1+error)
            X = evaluate_polynomial(poly, X)

    return X  


def delta_ortho(poly_degrees:List[int], b:float=1.99, max_step:int=100, delta:float=0.3, eps:float=1e-7) -> torch.Tensor:
    l,r = 0, b
    error = 1
    count = 0
    while abs(delta-error) > eps and count < max_step: #do first round,if doesn't fall in the range, then starts again with new boundaries that we know are too conservative
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
        count += 1
    a,b = 1-delta, 1+delta
    
    return polynomials, a , b  

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
    
    poly = [alpha, beta, gamma, delta]  
    return poly, error

from typing import Tuple, List
import numpy as np
import math

def c_n_k(n, k):
    # version robuste/entière
    return math.comb(n, k)


# ---------------------------------------------------------------------
# Kovarik (formule de secours si le système linéaire devient singulier)
# Zdislav Kovarik, "Some Iterative Methods for Improving Orthonormality", 1970
# ---------------------------------------------------------------------
def kovarik_formula(degree: int) -> np.poly1d:
    p = np.zeros(degree + 1, dtype=float)
    p[1] += 1.0  # coefficient de x^1
    a = 1.0
    for i in range(1, (degree + 1) // 2):
        for j in range(2 * (i - 1) + 1, 2 * i + 1):
            a *= j / 2
        a /= i ** 2
        sign = 1.0
        for k in range(0, i + 1):
            p[2 * k + 1] += a * sign * c_n_k(i, k)
            sign *= -1.0
    return np.poly1d(p[::-1])


_A, _B = None, None   

def get_polynomial(Ext: np.ndarray) -> np.ndarray:
    """
    Construit et résout le système d'équirépartition de l'erreur :
        sum_{k=1..n} a_k x^{2k-1} - 1 = s_i * E  sur les n+1 points 'Ext'
    où s_i = (-1)^i (alternance des signes).
    Retourne le vecteur c = [a_1, a_3, ..., a_{2n-1}, E].
    """
    n = len(Ext) - 1
    A = np.zeros((n + 1, n + 1), dtype=float)
    b = np.ones(n + 1, dtype=float)
    
    for i, x in enumerate(Ext):
        # colonnes pour x^1, x^3, ..., x^{2n-1}
        for k in range(n):
            A[i, k] = x ** (2 * k + 1)  # Fixed: should be 2k+1, not 2(k+1)-1
        # colonne de l'erreur alternée
        A[i, n] = (-1.0) ** i  # Fixed: cleaner alternating sign
    
    # résolution
    c = np.linalg.solve(A, b)
    return c  # longueur n+1


def get_ext(a_odd: np.ndarray) -> np.ndarray:
    """
    À partir des coefficients 'a_odd' (a1, a3, ..., a_{2n-1}),
    on construit p(x) = sum a_{2k-1} x^{2k-1}, on dérive p'(x),
    puis on récupère ses racines réelles dans l'intervalle (_A, _B).
    Ce sont (en pratique) les n-1 nouveaux points extrémaux internes.
    """
    n = len(a_odd)
    if n <= 1:
        return np.array([], dtype=float)

    # Coefficients du polynôme p en ordre décroissant pour np.poly1d
    # Taille 2n (degré max 2n-1). Les positions correspondant aux
    # puissances paires restent nulles, les impaires prennent a_odd.
    coeffs = np.zeros(2 * n, dtype=float)  # indices 0..2n-1 (deg 2n-1 down to 0)
    
    # pour k=0..n-1, terme a_odd[k] * x^{2k+1}
    # index dans 'coeffs' (ordre décroissant) pour x^{deg} est: idx = (2n-1) - deg
    for k in range(n):
        deg = 2 * k + 1
        idx = (2 * n - 1) - deg
        coeffs[idx] = a_odd[k]

    p = np.poly1d(coeffs)
    dp = np.polyder(p)

    roots = np.roots(dp)
    real_roots = roots[np.isclose(roots.imag, 0.0, atol=1e-12)].real
    
    # Fixed: consistent interval handling
    A_min, A_max = min(_A, _B), max(_A, _B)
    mask = (real_roots > A_min) & (real_roots < A_max)
    inside = real_roots[mask]
    inside.sort()
    
    # Return in the correct order for the interval
    if _A > _B:
        inside = inside[::-1]

    return inside


def build_polynomial_from_coeffs(a_odd: np.ndarray) -> np.poly1d:
    """
    Build polynomial from odd coefficients in consistent way.
    """
    n = len(a_odd)
    if n == 0:
        return np.poly1d([0.0])
    
    # Create coefficient array for polynomial (descending order)
    # Maximum degree is 2n-1
    coeffs = np.zeros(2 * n, dtype=float)
    
    # Fill in odd power coefficients
    for k in range(n):
        deg = 2 * k + 1
        idx = (2 * n - 1) - deg  # Convert to descending order index
        coeffs[idx] = a_odd[k]
    
    return np.poly1d(coeffs)


def remez_step(Ext: np.ndarray):
    """
    One step of the Remez exchange algorithm.
    """
    n = len(Ext) - 1
    c = get_polynomial(Ext)              
    a_odd = c[:n]
    
    # Fixed: use consistent polynomial construction
    p = build_polynomial_from_coeffs(a_odd)
    e = c[n].item()
    
    # Get new extremal points
    internal_points = get_ext(a_odd)
    NewExt = np.concatenate(([Ext[0]], internal_points, [Ext[-1]]))
    
    # Calculate new error
    f = lambda x: np.abs(p(x) - 1.0)
    newe = np.max(f(NewExt))
    
    return p, NewExt, e, newe


def remez_core(A: float, B: float, degree: int, max_iter: int = 100, tol: float = 1e-14):
    """
    Remez pour approximer f(x) = 1 sur [A,B] par un polynôme aux puissances impaires
    de degré <= degree (donc degré effectif 2n-1 avec n=(degree+1)//2).
    Retourne (poly1d, erreur_max).
    """
    global _A, _B
    _A, _B = A, B  # Keep original order for interval orientation

    if degree == 0:
        return np.poly1d([1.0]), 0.0

    n = (degree + 1) // 2

    # Initialize extremal points - ensure correct ordering
    if A <= B:
        Ext = np.linspace(A, B, n + 1, dtype=np.float64)
    else:
        Ext = np.linspace(A, B, n + 1, dtype=np.float64)
    
    p = np.poly1d([0.0])
    prev_error = float('inf')

    for iteration in range(max_iter):
        try:
            p, NewExt, e, newe = remez_step(Ext)
            
            # Fixed: proper convergence check
            # Check if we've converged (small relative improvement in error)
            if abs(newe - abs(e)) < tol * max(abs(e), newe) and iteration > 0:
                return p, float(newe)
            
            # Also check if error is not improving significantly
            if iteration > 0 and abs(newe - prev_error) < tol * prev_error:
                return p, float(newe)
                
            Ext = np.array(NewExt, dtype=np.float64)
            prev_error = newe
            
        except np.linalg.LinAlgError:
            # Fallback to Kovarik formula if system becomes singular
            print("Warning: Linear system singular, using Kovarik fallback")
            kovarik_poly = kovarik_formula(degree)
            # Calculate actual error for Kovarik polynomial
            test_points = np.linspace(min(A, B), max(A, B), 1000)
            kovarik_error = np.max(np.abs(kovarik_poly(test_points) - 1.0))
            return kovarik_poly, kovarik_error

    # If we reach here, we didn't converge within max_iter
    print(f"Muon-clip Warning: Remez algorithm did not converge within {max_iter} iterations with max error {newe}")
    return p, float(newe)


def remez(s_interval: Tuple[float, float], poly_degree: int, 
          max_iter: int = 100, tol: float = 1e-14) -> Tuple[List[float], float]:
    """
    Compute the Remez coefficients (minimax) to approximate f(x)=1 on [A,B]
    with an odd-power polynomial subspace (x, x^3, ..., x^{2n-1}).

    Args:
        s_interval: (A, B) interval
        poly_degree: requested degree bound (the effective degree becomes 2n-1 with n=(degree+1)//2)
        max_iter: maximum number of iterations
        tol: convergence tolerance

    Returns:
        Tuple[List[float], float]: (coefficients of the polynomial in descending order, max_error)
    """
    A, B = s_interval
    p, err = remez_core(A, B, poly_degree, max_iter, tol)
    # np.poly1d.c already returns coefficients in descending order 
    return p.c.astype(float).tolist(), err

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

def newton_schulz_accelerated(G:torch.Tensor,  ortho_polynomials, poly_degree:List[float]=None, order:int=3, estimate_lower_bound:bool=True, lower_bound=1e-3, iter:int=9) -> torch.Tensor:

    G, A = gelfand_upper_bound(G)
    if estimate_lower_bound: s_interval = (None, 1)
    else: s_interval = (lower_bound, 1)

    if poly_degree is None:
        poly_degree = [order] * iter
    
    X = cans_ortho(G, s_interval,ortho_polynomials, poly_degree)
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


def muon_update(grad, momentum, velocity, eps, step, ortho_polynomials, beta:float=0.95, ns_steps:int=5, nesterov:bool=True, better_ortho:bool=False):
    if grad.ndim != 2:
        raise ValueError(f"Input tensor must be 2D but got shape {grad.shape}")
    
    momentum.lerp_(grad, 1 - beta)
    grad = grad.lerp_(momentum, beta) if nesterov else momentum
    if grad.ndim == 4: # for the case of conv filters
        grad = grad.view(len(grad), -1)

    is_transpose = grad.size(0) > grad.size(1)
    if is_transpose: grad = grad.T  
    if better_ortho: grad = newton_schulz_accelerated(grad, iter=ns_steps, ortho_polynomials=ortho_polynomials)
    else: grad = newtonschulz(grad, steps=ns_steps)
    grad *= max(1, grad.size(-2) / grad.size(-1))**0.5
    if is_transpose: grad = grad.T
    
    velocity.lerp_(torch.linalg.vector_norm(grad,dim=-1,keepdim=True),1-beta)
    velocity=torch.mul(velocity,1/(1-beta**step))
    grad = torch.mul(grad,1/torch.sqrt(velocity+eps))

    return grad






