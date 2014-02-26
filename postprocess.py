import numpy as np
import posterior as pos
import scipy.stats as ss

def true_density(p, z, zmax=0.1):
    lc, lb, mu, sigma, A = p

    return (lc*ss.norm.pdf(z, loc=mu, scale=sigma) + 2.0*lb*(A*(zmax-z)/(zmax*zmax) + (1.0-A)*z/(zmax*zmax)))/(lc+lb)
