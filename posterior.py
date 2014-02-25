import numpy as np
import scipy.special as sp
import scipy.stats as ss

def draw_data(Rc, Rb, muz, sigmaz, A, mu_noise = np.log(0.01), sigma_noise = 0.1, zmax=0.1):
    Nc = np.random.poisson(Rc)
    Nb = np.random.poisson(Rb)

    czs = muz + sigmaz*np.random.randn(Nc)
    cdzs = np.random.lognormal(mean=mu_noise, sigma=sigma_noise, size=Nc)

    us = np.random.uniform(size=Nb)
    bzs = zmax/(2.0*A-1.0)*(A - np.sqrt(A*A - 2.0*A*us + us))
    bdzs = np.random.lognormal(mean=mu_noise, sigma=sigma_noise, size=Nb)

    zs = np.concatenate((czs, bzs), axis=0)
    dzs = np.concatenate((cdzs, bdzs), axis=0)

    data = np.column_stack((zs, dzs))
    data = np.random.permutation(data)
    data[:,0] += data[:,1]*np.random.randn(data.shape[0])

    return data

class Posterior(object):
    def __init__(self, zs, dzs, zmax=0.1):
        self._zs = zs
        self._dzs = dzs
        self._zmax = zmax

    @property
    def zs(self):
        return self._zs

    @property
    def dzs(self):
        return self._dzs

    @property
    def zmax(self):
        return self._zmax

    @property
    def dtype(self):
        return np.dtype([('Rc', np.float),
                         ('Rb', np.float),
                         ('muz', np.float),
                         ('sigmaz', np.float),
                         ('A', np.float)])

    def to_params(self, p):
        return p.view(self.dtype)

    def log_convolved_foreground_density(self, mu, sigma):
        sigmas = np.sqrt(sigma*sigma + self.dzs*self.dzs)
        return ss.norm.logpdf(self.zs, loc=mu, scale=sigmas)

    def log_convolved_background_density(self, A):
        zmax = self.zmax
        zmax2 = zmax*zmax

        zs = self.zs
        dzs = self.dzs

        term1 = (A*zmax - 2.0*A*zs + zs)*(sp.erf((zmax - zs)/(np.sqrt(2.0)*dzs)) + sp.erf(zs/(np.sqrt(2.0)*dzs)))/zmax2

        term2 = np.sqrt(2.0)*dzs*(2.0*A-1.0)/(np.sqrt(np.pi)*zmax2)*(np.exp(-0.5*np.square(zmax-zs)/np.square(dzs)) - np.exp(-0.5*np.square(zs)/np.square(dzs)))

        return np.log(term1 + term2)
        
    def log_prior(self, p):
        p = self.to_params(p)

        Rc = p['Rc']
        Rb = p['Rb']
        muz = p['muz']
        sigmaz = p['sigmaz']
        A = p['A']

        if Rc < 0 or Rb < 0:
            return np.NINF

        if sigmaz < 0:
            return np.NINF

        if A < 0 or A > 1:
            return np.NINF

        if muz - 2.0*sigmaz < 0 or muz + 2.0*sigmaz > self.zmax:
            return np.NINF

        if np.abs(A-0.5) > 1e-2:
            A2m1 = 2.0*A - 1.0
            Aprior2 = -(2.0*(4.0*A - np.log(A) + np.log1p(-A) - 2.0))/(A2m1*A2m1*A2m1)
            Aprior = np.sqrt(Aprior2)
        else:
            Aprior = 2.0*np.sqrt(3.0)/3.0 + 4.0/5.0*np.sqrt(3.0)*np.square(A-0.5)

        eps = np.mean(self.dzs)
        sigprior2 = sigmaz / (eps*eps + sigmaz*sigmaz)

        return -0.5*(np.log(Rc) + np.log(Rb)) + np.log(sigprior2) + np.log(Aprior)

    def log_likelihood(self, p):
        p = self.to_params(p)

        Rc = p['Rc']
        Rb = p['Rb']
        muz = p['muz']
        sigmaz = p['sigmaz']
        A = p['A']

        log_pfore = self.log_convolved_foreground_density(muz, sigmaz)
        log_pback = self.log_convolved_background_density(A)

        log_likes = np.logaddexp(np.log(Rc) + log_pfore,
                                 np.log(Rb) + log_pback)

        return np.sum(log_likes) - Rc - Rb

    def __call__(self, p):
        lp = self.log_prior(p)

        if lp == np.NINF:
            return np.NINF
        else:
            ll = self.log_likelihood(p)
            if np.isnan(ll):
                print p
                raise ValueError("ll = NaN")
            return lp + ll
