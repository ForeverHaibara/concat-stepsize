"""
Implementation of PriDP and DomDp (Algorithms 4.1 and 4.2).
"""
from math import sqrt
phi = lambda x, y: (-x - y + sqrt((x + y + 2)**2 + 4*(x + 1)*(y + 1)))/2
psi = lambda x, y: (3 - 2*y + sqrt((2*y + 1)*(2*y + 8*x + 9)))/4

def PriDP(N):
    """
    Algorithm 4.1
    NOTE: the inner for-loop can be vectorized using NumPy,
    but here we only present the naive implementation.
    """
    H1, S1 = [[]], [0]
    for n in range(1, N+1):
        k_best, s_max = 0, 0
        for k in range(n):
            k2 = n - k - 1
            s = S1[k] + phi(S1[k], S1[k2]) + S1[k2]
            if s >= s_max:
                k_best, s_max = k, s

        k, k2 = k_best, n - k_best - 1
        H1.append(H1[k] + [phi(S1[k], S1[k2])] + H1[k2])
        S1.append(s_max)
    return H1, S1
    
def DomDP(N):
    """
    Algorithm 4.2
    NOTE: the inner for-loop can be vectorized using NumPy,
    but here we only present the naive implementation.
    """
    H2, S2 = [[]], [0]
    H1, S1 = PriDP(N)
    for n in range(1, N+1):
        k_best, s_max = 0, 0
        for k in range(n):
            k2 = n - k - 1
            s = S1[k] + psi(S1[k], S2[k2]) + S2[k2]
            if s >= s_max:
                k_best, s_max = k, s

        k, k2 = k_best, n - k_best - 1
        H2.append(H1[k] + [psi(S1[k], S2[k2])] + H2[k2])
        S2.append(s_max)
    return H2, S2

def verify_bound(h, bound = 'f', L = 1, verbose = True, **kwargs):
    """
    Numerically verify a primitve/dominant stepsize h satisfies
    
        f_n - f_* <= 1/(2(1^Th) + 1) * L/2 * ||x_0 - x_*||^2.

    or verify a g-bounded stepsize schedule h satisfies
    
        ||g_n||^2 <= 1/(2(1^Th) + 1) * (2*L) * (f_0 - f_*)

    The verification relies on the PEPit package [1].

    [1] B. Goujaud, C. Moucer, F. Glineur, J. Hendrickx, A. Taylor, A. Dieuleveut.
    "PEPit: computer-assisted worst-case analyses of first-order optimization methods in Python."
    Math. Prog. Comp. 16, 337-367 (2024). https://doi.org/10.1007/s12532-024-00259-7
    """
    try:
        from PEPit import PEP
        from PEPit.functions import SmoothConvexFunction
    except ImportError as e:
        print('PEPit is not installed. Skipping verification.')
        return

    # the code below is adapted from the example of PEPit
    problem = PEP()

    func = problem.declare_function(SmoothConvexFunction, L=L)
    xs = func.stationary_point()
    fs = func(xs)
    x0 = problem.set_initial_point()
    x = x0
    for hi in h:
        x = x - hi/L * func.gradient(x)
        
    if bound == 'f':
        problem.set_initial_condition((x0 - xs) ** 2 <= 1)
        problem.set_performance_metric(func(x) - fs)    

        theoretical_tau = L/2 / (2 * sum(h) + 1)
        summary = lambda x: "f(x_n)-f_* <= {:.6} ||x_0 - x_*||^2".format(x)
    elif bound == 'g':
        problem.set_initial_condition(func(x0) - fs <= 1)
        problem.set_performance_metric(func.gradient(x)**2)

        theoretical_tau = (2*L) / (2 * sum(h) + 1)
        summary = lambda x: "||g_n||^2 <= {:.6} (f_0 - f_*)".format(x)
    else:
        raise ValueError('bound must be either "f" or "g"')        

    # Solve the PEP
    pepit_tau = problem.solve(**kwargs)

    # Compute theoretical guarantee (for comparison)
    if verbose:
        print('Worst-case performance of gradient descent with fixed step-sizes')
        print(f'\tPEPit       guarantee:\t {summary(pepit_tau)}')
        print(f'\tTheoretical guarantee:\t {summary(theoretical_tau)}')
        print('Numerical error: {}'.format(abs(pepit_tau - theoretical_tau)))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    ######################################################################
    # Print the stepsize schedules
    ######################################################################
    N = 10
    H, S = DomDP(N)
    for i in range(N+1):
        print('Stepsize schedule', i, ':', H[i])

    ######################################################################
    # Verify the worst-case bound of the last stepsize schedule
    ######################################################################
    h = H[N]
    print("=" * 50 + "\nVerifying the last stepsize schedule...")
    verify_bound(h, bound='f', L=1, verbose=True)

    print("=" * 50 + "\nVerifying the reverse of the last stepsize schedule...")
    verify_bound(h[::-1], bound='g', L=1, verbose=True)
