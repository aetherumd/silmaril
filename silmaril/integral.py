import numpy as np

def compute_lumonosity(lowerBound, upperBound, lamd_datapoints, T_datapoints):

    lamd_func, T_func = make_vectors_similar(lamd_datapoints, T_datapoints)

    # make new function
    top_integrand = lambda x: np.dot(lamd_func(x), T_func(x), x)
    bottom_integrand = lambda x: np.dot(x, lamd_func(x))
    
    
    filter_flux = integrate_function(top_integrand, lowerBound, upperBound)
    flux = integrate_function(bottom_integrand, lowerBound, upperBound)

    return filter_flux / flux

def make_vectors_similar(vectorA, vectorB):
    pass

def integrate_function(func, a, b, num_points=1000):
    """
    Compute the definite integral of a function using the trapezoidal rule.
    
    Parameters:
    -----------
    func : callable
        The function to integrate
    a : float
        Lower bound of integration
    b : float
        Upper bound of integration
    num_points : int, optional
        Number of points to use (default: 1000)
    
    Returns:
    --------
    float
        The approximate value of the definite integral
    """
    x = np.linspace(a, b, num_points)
    y = func(x)
    return np.trapz(y, x)