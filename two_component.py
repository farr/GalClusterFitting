import numpy as np
import scipy.stats as ss

def log_inv_wishart(x, nu, y):
    r"""Log of the PDF of the inverse Wishart distribution.

    :param x: The matrix at which to evaluate the PDF.

    :param nu: The DOF parameter for the distribution.  Note
      :math:`\nu > n-1` where :math:`n` is the dimension of the
      matrices ``x`` and ``y``.

    :param y: The shape matrix for inverse Wishart.

    The PDF is 

    ..math::

    \mathcal{W}(x, \nu, y) = \frac{\left| y \right|^{\nu/2}}{2^{\nu n/2} \Gamma_n\left( \frac{\nu}{2}\right)} |x|^{-\frac{\nu + n + 1}{2}} \exp\left[ -\frac{1}{2} \tr \left( y x^{-1} \right) \right]

    """
    s, ldx = np.linalg.slogdet(x)
    s, ldy = np.linalg.slogdet(y)

    tr_prod = np.sum(np.diag(np.linalg.solve(x, y)))

    return nu/2.0*ldy - (nu + x.shape[0]+1)/2.0*ldx - 0.5*tr_prod

def log_multinormal(x, mu, sigma):
    """Returns the value of the multivariate normal PDF with mean(s)
    ``mu`` and covariance(s) ``sigma`` at the points ``x``.  Allows
    for multiple means and covariances, in which case it will return
    multiple arrays of PDF values.

    :param x: Array of shape ``(M, N)`` giving the M points of
      dimension N at which to evaluate the PDF.

    :param mu: Array of shape ``(N,)`` giving the mean of the
      distribution.

    :param sigma: Array of shape ``(M, N, N)`` giving the coviariances
      associated with each point.

    :return: Array of shape ``(M,)`` giving the values of multivariate
      normal PDF at the ``M`` points.

    """

    x = np.atleast_2d(x)

    s, lds = np.linalg.slogdet(sigma)

    dx = x - mu

    return -0.5*(np.log(2.0*np.pi) + lds) - 0.5*np.sum(dx*np.linalg.solve(sigma, dx), axis=1)

class TwoComponent(object):
    """A posterior object representing a two-component Gaussian mixture
    model for the galaxies in a field containing a group.  The model
    allows for different errors for each measured redshift.

    """

    def __init__(self, zs, dzs, ras, decs, z0, dz0, ra0, dec0):
        """Initialise the posterior.  

        :param zs: The mesaured redshifts.

        :param dzs: The redshift error.

        :param ras: The measured RAs in decimal degrees.

        :param decs: The measured DECs in decimal degrees.

        :param z0: The identified cluster centroid.

        :param dz0: The initial guess at the redshift size of the
          cluster.

        :param ra0: The RA of the guessed centroid, in decimal degrees.

        :param dec0: The DEC of the guessed centroid, in decimal degrees.

        """

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

    @property
    def pnames(self):
        return [r'$A$', 
                r'$\mu_c^\alpha$', r'$\mu_c^\delta$', r'$\mu_c^z$',
                r'$\mu_b^\alpha$', r'$\mu_b^\delta$', r'$\mu_b^z$',
                r'\Sigma_c^{\alpha \alpha}$', r'$\Sigma_c^{\alpha\delta}$', r'$\Sigma_c^{\alpha z}$',
                r'$\Sigma_c^{\delta \delta}$', r'$\Sigma_c^{\delta z}$', r'$\Sigma_c^{z z}$',
                r'\Sigma_b^{\alpha \alpha}$', r'$\Sigma_b^{\alpha\delta}$', r'$\Sigma_b^{\alpha z}$',
                r'$\Sigma_b^{\delta \delta}$', r'$\Sigma_b^{\delta z}$', r'$\Sigma_b^{z z}$']

    def to_params(self, p):
        """Returns a named view of the array ``p`` using the params dtype.

        """
        return np.atleast_1d(p).view(self.dtype).squeeze()

    def covariance_matrices(self, p):
        """Returns ``(cov_cluster, cov_background)``, the covariance matrices
        of the cluster and background components in the model for the
        parameters ``p``.

        """

        p = self.to_params(p)

        covc = self._cov_matrix(p['covc'])
        covb = self._cov_matrix(p['covb'])

        return (covc, covb)

    def _cov_matrix(self, x):
        r"""Returns the 3x3 covariance matrix associated with parameters ``x``.
        The parameterisation we use is the same as Stan (see
        http://mc-stan.org/ ).  If :math:`\Sigma` is the covariance
        matrix, then we have 

        .. math::

          \Sigma = z z^T,

        with :math:`z` lower triangular, derived from a lower
        triangular matrix :math:`y` whose entries off-diagonal match
        :math:`z`, but whose diagional entries are the log of those of
        :math:`z`:
        
        .. math::

          \diag(y) = \log(\diag(z))

        The matrix :math:`y` is filled in, in ``np.tril_indices``
        order from ``x``.

        """

        y = np.zeros((3,3))
        y[np.tril_indices(3)] = x
        y.flatten()[::4] = np.exp(y.flatten()[::4])

        return np.dot(y, y.T)

    def _cov_matrix_reverse(self, cm):
        """Returns the ``x`` such that ``self._cov_matrix(x) == cm``.

        """

        l = np.linalg.cholesky(cm)

        l.flatten()[::4] = np.log(l.flatten()[::4])
        return l[np.tril_indices(3)]

    def _cov_matrix_log_jac(self, x):
        """Returns the log of the jacobian factor for the relationship between
        a density in covariance-matrix space and in ``x``-space for
        the parameterisation used in ``_cov_matrix`` above.

        """

        y = np.zeros((3,3))
        y[np.tril_indices(3)] = x
        log_dy = np.diag(y) # log_dy contains the log of the diagonal elements of z
        expts = 3 - np.arange(1, 4) + 2

        return np.dot(expts, log_dy)

    def log_prior(self, p):
        r"""Returns the log of the prior for the given parameters, ``p``.  

        The prior is inverse Wishart with :math:`\nu = 3` for the
        covariance matrices of the two components.  The background
        component uses a scale matrix equal to the covariance of the
        entire set of points, while the group scale matrix is fixed at
        ``np.diag([0.25, 0.25, dz0*dz0])``.

        The prior is Jeffreys for the weight parameter:

        ..math::

          p(A) = \frac{1}{\sqrt{A(1-A)}}

        For the mean parameters, the prior is normal.  The background
        mean's prior has center equal to the mean of all the points
        and covariance equal to the variance of the points in each
        dimension.  The group prior uses the identified centroid and
        has covariance equal to ``np.diag([0.25, 0.25, dz0*dz0])``.

        """

        p = self.to_params(p)

        covc = self._cov_matrix(p['covc'])
        covb = self._cov_matrix(p['covb'])

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

        lp += self._cov_matrix_log_jac(p['covc'])
        lp += self._cov_matrix_log_jac(p['covb'])

        return lp

    def _log_foreground(self, p):
        """Returns the log of the foreground density at the indicated points.

        """
        p = self.to_params(p)
        
        covc = self._cov_matrix(p['covc'])
        covcs = np.tile(covc, (self.pts.shape[0], 1, 1))
        covcs[:,2,2] += self.dzs*self.dzs

        return log_multinormal(self.pts, p['muc'], covcs)
        
    def _log_background(self, p):
        """Returns the log of the background density at the indicated points.q

        """
        p = self.to_params(p)

        covb = self._cov_matrix(p['covb'])
        covbs = np.tile(covb, (self.pts.shape[0], 1, 1))
        covbs[:,2,2] += self.dzs*self.dzs

        return log_multinormal(self.pts, p['mub'], covbs)

    def log_likelihood(self, p):
        """Returns the log-likelihood at the parameters ``p``.

        """

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

    def _best_guess(self):
        p = self.to_params(np.zeros(self.nparams))

        p['A'] = np.float(np.sum(np.abs(self.zs - self.z0) < 2.0*self.dz0))/self.zs.shape[0]

        p['muc'] = self.cmean
        p['mub'] = self.mean

        p['covc'] = self._cov_matrix_reverse(self.cvar)
        p['covb'] = self._cov_matrix_reverse(self.var)

        return p

    def pguess(self, p0=None):
        """Returns a draw from an approximate posterior assuming that the true
        parameters are near ``p0``.  If no ``p0`` is given, then
        produce a guess at the true parameters.

        """
        if p0 is None:
            p0 = self._best_guess()
        else:
            p0 = self.to_params(p0)
        pguess = p0.copy()

        factor = 7
        factor2 = factor*factor

        N = self.pts.shape[0]

        covc = self._cov_matrix(p0['covc'])
        covb = self._cov_matrix(p0['covb'])

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
                covc_guess[j,i] = covc_guess[i,j]
                covb_guess[i,j] = np.random.normal(scale=np.sqrt(covb_guess[i,i]*covb_guess[j,j]/(factor2*N)))
                covb_guess[j,i] = covb_guess[i,j]

        pguess['covc'] = self._cov_matrix_reverse(covc_guess)
        pguess['covb'] = self._cov_matrix_reverse(covb_guess)

        return pguess.reshape((1,)).view(float)

    def log_pfore(self, p):
        """Returns the log of the foreground probability for each galaxy
        sample given parameters ``p``.

        """

        p = self.to_params(p)

        lc = np.log(p['A']) + self._log_foreground(p)
        lb = np.log1p(-p['A']) + self._log_background(p)

        return lc - np.logaddexp(lc, lb)

    def cluster_rs(self, p):
        """Returns the two correlation coefficients between the principal axes
        of the foreground distribution on the sky and the redshift
        dimension given the parameters ``p``.

        """

        p = self.to_params(p)

        covc = self._cov_matrix(p['covc'])

        evals, evecs = np.linalg.eigh(covc[:2,:2])
        evecs = evecs[:, np.argsort(evals)[::-1]]
        
        e = np.eye(3)
        e[:2,:2] = evecs

        ecov = np.dot(e.T, np.dot(covc, e))

        rs = np.array([ecov[0,2]/np.sqrt(ecov[0,0]*ecov[2,2]),
                       ecov[1,2]/np.sqrt(ecov[1,1]*ecov[2,2])])

        return rs
        
    def log_rho_sky(self, p, pts):
        """Returns the log of the number density (per square degree) on the
        sky for the group foreground given parameters ``p`` at the
        points ``pts``.

        """

        p = self.to_params(p)

        N = np.random.gamma(self.pts.shape[0]+0.5)

        mu = p['muc'][:2]
        cov = self._cov_matrix(p['covc'])[:2,:2]

        dxs = pts - mu

        log_norm = -0.5*(np.log(2.0*np.pi) + np.linalg.slogdet(cov)[1]) + np.log(N) + np.log(p['A']) - np.log(np.cos(pts[:,1]*np.pi/180.0))

        return log_norm - 0.5*np.sum(dxs*np.linalg.solve(cov, dxs.T).T, axis=1)

    def cluster_size(self, p):
        """Returns the standard deviation, ``[sigma_ra, sigma_dec, sigma_z]``,
        of the cluster component associated with parameters ``p``.

        """
        p = self.to_params(p)
        covc = self._cov_matrix(p['covc'])
        return np.sqrt(np.diag(covc))

    def ellipticity(self, p):
        """Returns the ellipticity of the cluster component on the sky for the
        parameters ``p``.

        """

        p = self.to_params(p)

        ccov = self._cov_matrix(p['covc'])

        evals = np.linalg.eigvalsh(ccov[:2,:2])

        return np.sqrt(np.min(evals)/np.max(evals))
