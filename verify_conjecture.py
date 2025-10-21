"""
Verification of Conjecture in Subsection: Nonuniqueness for n <= 10^4
"""
from math import sqrt

S = [0]

phi = lambda x,y: (- x - y + sqrt((x+y+2)**2 + 4*(x+1)*(y+1)))/2

# compute 1^TConPP(h_a, h_b) given 1^Th_a an 1^Th_b
concat = lambda x, y: x + phi(x, y) + y
# concat = lambda x, y: (x + y + sqrt((x+y+2)**2 + 4*(x+1)*(y+1)))/2


def core(n):
    """Return p, l such that n = p * 2^l where p is odd"""
    l = 0
    while n > 0 and n % 2 == 0:
        n //= 2
        l += 1
    return n, l

if __name__== '__main__':
    N = 10000
    err = 0
    for n in range(1, N+1):
        s = 0
        for k in range(n):
            s_k = concat(S[k], S[n-k-1])
            s = max(s, s_k)

        s_k = concat(S[(n-1)//2], S[n//2])
        err = max(err, abs(s_k - s))

        if n % 2 == 1:
            p, l = core(n+1)
            if p > 1 and p % 2 == 1:
                a = (p-1)*2**(l-1) - 1
                b = (p+1)*2**(l-1) - 1
                s_k = concat(S[a], S[b])
                err = max(err, abs(s_k - s))

        if n % 1000 == 0:
            print(f'Verified n <= {n}\nMaximum error = {err} To be continued...')
        S.append(s)

    print(f'Verification Complete.\nN = {n}\nMaximum error = {err}')
