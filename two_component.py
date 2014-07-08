import numpy as np
import scipy.stats as ss

def log_inv_wishart(x, nu, y):
    s, ldx = np.linalg.slogdet(x)
    s, ldy = np.linalg.slogdet(y)

    tr_prod = np.sum(np.diag(np.linalg.solve(x, y)))

    return nu/2.0*ldy - (nu + x.shape[0]+1)/2.0*ldx - 0.5*tr_prod

def log_multinormal_old(x, mu, sigma):
    s, lds = np.linalg.slogdet(sigma)
    
    dx = x - mu

    return -0.5*(np.log(2.0*np.pi) + lds) - 0.5*np.dot(dx, np.linalg.solve(sigma, dx))

def log_multinormal(x, mu, sigma):
    x = np.atleast_2d(x)

    s, lds = np.linalg.slogdet(sigma)

    dx = x - mu

    return -0.5*(np.log(2.0*np.pi) + lds) - 0.5*np.sum(dx*np.linalg.solve(sigma, dx), axis=1)

def upper_tri_to_matrix(ut, dim):
    m = np.zeros((dim,dim))
    m[np.triu_indices(dim)] = ut
    m = m + m.T
    for i in range(dim):
        m[i,i] /= 2.0
    return m

class TwoComponent(object):
    def __init__(self, zs, dzs, ras, decs, z0, dz0, ra0, dec0):
        self.zs = zs
        self.dzs = dzs
        self.ras = ras
        self.decs = decs
        self.z0 = z0
        self.dz0 = dz0
        self.ra0 = ra0
        self.dec0 = dec0

        self.pts = np.column_stack((self.ras, self.decs, self.zs))
        self.mean = np.mean(self.pts, axis=0)
        self.var = np.diag(np.var(self.pts, axis=0))
        self.sigma = np.sqrt(self.var)
        
        self.cmean = np.array([self.ra0, self.dec0, self.z0])
        self.cvar = np.diag([0.25, 0.25, self.dz0*self.dz0])
        self.csigma = np.sqrt(self.cvar)

    @property
    def nparams(self):
        return 19

    @property
    def dtype(self):
        return np.dtype([('A', np.float),
                         ('muc', np.float, 3),
                         ('mub', np.float, 3),
                         ('covc', np.float, 6),
                         ('covb', np.float, 6)])

    def to_params(self, p):
        return np.atleast_1d(p).view(self.dtype).squeeze()

    def log_prior(self, p):
        p = self.to_params(p)

        covc = upper_tri_to_matrix(p['covc'], 3)
        covb = upper_tri_to_matrix(p['covb'], 3)

        if p['A'] < 0 or p['A'] > 1:
            return np.NINF
        if np.any(np.linalg.eigvalsh(covc) <= 0.0):
            return np.NINF
        if np.any(np.linalg.eigvalsh(covb) <= 0.0):
            return np.NINF

        lp = 0.0

        lp += -0.5*(np.log(p['A']) + np.log1p(-p['A']))

        lp += np.sum(ss.norm.logpdf(p['muc'], loc=self.cmean, scale=np.diag(self.csigma)))
        lp += np.sum(ss.norm.logpdf(p['mub'], loc=self.mean, scale=np.diag(self.sigma)))

        lp += log_inv_wishart(covc, 3, self.cvar)
        lp += log_inv_wishart(covb, 3, self.var)

        return lp

    def _log_foreground(self, p):
        p = self.to_params(p)
        
        covc = upper_tri_to_matrix(p['covc'], 3)
        covcs = np.tile(covc, (self.pts.shape[0], 1, 1))
        covcs[:,2,2] += self.dzs*self.dzs

        return log_multinormal(self.pts, p['muc'], covcs)
        
    def _log_background(self, p):
        p = self.to_params(p)

        covb = upper_tri_to_matrix(p['covb'], 3)
        covbs = np.tile(covb, (self.pts.shape[0], 1, 1))
        covbs[:,2,2] += self.dzs*self.dzs

        return log_multinormal(self.pts, p['mub'], covbs)

    def log_likelihood(self, p):
        p = self.to_params(p)

        lc = self._log_foreground(p)
        lb = self._log_background(p)

        return np.sum(np.logaddexp(np.log(p['A']) + lc,
                                   np.log1p(-p['A']) + lb))

    def __call__(self, p):
        lp = self.log_prior(p)
        
        if lp == np.NINF:
            return lp
        else:
            return lp + self.log_likelihood(p)

    def pguess(self, p0):
        p0 = self.to_params(p0)
        pguess = p0.copy()

        factor = 7
        factor2 = factor*factor

        N = self.pts.shape[0]

        covc = upper_tri_to_matrix(p0['covc'], 3)
        covb = upper_tri_to_matrix(p0['covb'], 3)

        ncl = p0['A']*self.pts.shape[0]
        pguess['A'] = np.random.beta(ncl*factor2+1.0, (N-ncl)*factor2+1.0)

        pguess['muc'] = np.random.multivariate_normal(p0['muc'], covc/(factor2*N))
        pguess['mub'] = np.random.multivariate_normal(p0['mub'], covb/(factor2*N))

        covc_guess = covc.copy()
        covb_guess = covb.copy()

        for i in range(3):
            covc_guess[i,i] = np.random.lognormal(mean=np.log(covc[i,i]), sigma=1.0/(factor*np.sqrt(N)))
            covb_guess[i,i] = np.random.lognormal(mean=np.log(covb[i,i]), sigma=1.0/(factor*np.sqrt(N)))

        for i in range(3):
            for j in range(i+1, 3):
                covc_guess[i,j] = np.random.normal(scale=np.sqrt(covc_guess[i,i]*covc_guess[j,j]/(factor2*N)))
                covb_guess[i,j] = np.random.normal(scale=np.sqrt(covb_guess[i,i]*covb_guess[j,j]/(factor2*N)))

        pguess['covc'] = covc_guess[np.triu_indices(3)]
        pguess['covb'] = covb_guess[np.triu_indices(3)]

        return pguess.reshape((1,)).view(float)

    def draw(self, p):
        p = self.to_params(p)

        covc = upper_tri_to_matrix(p['covc'], 3)
        covb = upper_tri_to_matrix(p['covb'], 3)

        bpts = np.random.multivariate_normal(p['mub'], covb, size=self.pts.shape[0])
        cpts = np.random.multivariate_normal(p['muc'], covc, size=self.pts.shape[0])

        us = np.random.uniform(low=0, high=1, size=self.pts.shape[0])

        pts = np.zeros((self.pts.shape[0], 3))
        pts[us<p['A'], :] = cpts[us<p['A'], :]
        pts[us>=p['A'], :] = bpts[us>=p['A'], :]

        pts[:,2] += np.random.normal(scale=self.dzs)

        return pts

    def log_pfore(self, p):
        p = self.to_params(p)

        lc = np.log(p['A']) + self._log_foreground(p)
        lb = np.log1p(-p['A']) + self._log_background(p)

        return lc - np.logaddexp(lc, lb)

    def cluster_rs(self, p):
        p = self.to_params(p)

        covc = upper_tri_to_matrix(p['covc'], 3)

        rs = []
        for i in range(0,3):
            for j in range(i+1,3):
                rs.append(covc[i,j]/np.sqrt(covc[i,i]*covc[j,j]))
        rs = np.array(rs)

        return rs
        
