import numpy as np
import scipy.special as sp
import scipy.stats as ss

def draw_data(Rc, Rb, muz, sigmaz, A, mu_noise = np.log(0.01), sigma_noise = 0.1, zmin=0.0, zmax=0.1):
    r"""Draws data according to the model.  

    The model used is the following:

    For the cluster galaxies, 

    .. math::
    
      z_\mathrm{true} \sim N\left[ \mu_z, \sigma_z \right]

    while for the background galaxies, 

    .. math::

      z_\mathrm{true} \sim 2 \left( \frac{A}{z_\mathrm{max}^2} \left(z_\mathrm{max} - z \right) + \frac{1-A}{z_\mathrm{max}^2} z \right)

    (This is just a normalised, linear density profile, with
    :math:`p(z=0) = A/z_mathrm{max}`.)  For all galixes, the observed
    redshift includes a Gaussian error

    .. math::

      p\left( z_\mathrm{obs} | z_\mathrm{true} \right) = N\left[ 0, \sigma_n \right] \left( z_\mathrm{obs} - z_\mathrm{true} \right)

    :param Rc: The Poisson mean number of cluster galaxies.

    :param Rb: The Poisson mean number of background galaxies.

    :param muz: The mean true redshift of the cluster galaxies.

    :param sigmaz: The standard deviation of the cluster galaxy
      redshifts.

    :param A: The probability density at redshift 0 of the background
      galaxies.

    :param mu_noise: The mean parameter of the log-normal distribution
      from which the noise sigmas are drawn.

    :param sigma_noise: The sigma parameter of the log-normal
      distribution from which the noise sigmas are drawn.

    :param zmin: The minimum true redshift for the background.

    :param zmax: The maximum true redshift for the background.

    """
    Nc = np.random.poisson(Rc)
    Nb = np.random.poisson(Rb)

    czs = muz + sigmaz*np.random.randn(Nc)
    cdzs = np.random.lognormal(mean=mu_noise, sigma=sigma_noise, size=Nc)

    us = np.random.uniform(size=Nb)
    bzs = (A*zmax + (A-1)*zmin - np.sqrt(np.square(zmax-zmin)*(A*A - 2.0*A*us + us)))/(2.0*A - 1.0)
    bdzs = np.random.lognormal(mean=mu_noise, sigma=sigma_noise, size=Nb)

    zs = np.concatenate((czs, bzs), axis=0)
    dzs = np.concatenate((cdzs, bdzs), axis=0)

    data = np.column_stack((zs, dzs))
    data = np.random.permutation(data)
    data[:,0] += data[:,1]*np.random.randn(data.shape[0])

    return data

parameter_labels = [r'$\Lambda_\mathrm{cluster}$', r'$\Lambda_\mathrm{bground}$',
                    r'$\mu_z$', r'$\sigma_z$', r'$A$']

class ZPosterior(object):
    """Callable object representing the posterior.

    """
    
    def __init__(self, zs, dzs, zmin = 0.0, zmax=0.1):
        """Initialize the posterior with the given redshifts and
        uncertainties.

        """
        self._zs = zs
        self._dzs = dzs
        self._zmax = zmax
        self._zmin = zmin

    @property
    def zs(self):
        """The galaxy redshifts."""
        return self._zs

    @property
    def dzs(self):
        """Galaxy redshfit uncertainty."""
        return self._dzs

    @property
    def zmin(self):
        """The minimum true redshift of a background galaxy."""
        return self._zmin

    @property
    def zmax(self):
        """The maximum true redshift of a galaxy."""
        return self._zmax

    @property
    def dtype(self):
        """Data type describing the parameters of the model."""
        return np.dtype([('Rc', np.float),
                         ('Rb', np.float),
                         ('muz', np.float),
                         ('sigmaz', np.float),
                         ('A', np.float)])

    def to_params(self, p):
        """Returns a view of ``p`` in the parameter data type."""
        return p.view(self.dtype)

    def log_convolved_foreground_density(self, mu, sigma):
        """The log of the foreground density convolved with the observational
        errors.
        """
        sigmas = np.sqrt(sigma*sigma + self.dzs*self.dzs)
        return ss.norm.logpdf(self.zs, loc=mu, scale=sigmas)

    def log_convolved_background_density(self, A):
        """The log of the background density convolved with the observational
        errors.

        """
        zmax = self.zmax
        zmin = self.zmin

        deltaz = zmax - zmin
        deltaz2 = deltaz*deltaz

        zs = self.zs
        dzs = self.dzs

        dzs2 = dzs*dzs

        edenoms = 1.0/(np.sqrt(2.0)*dzs)

        norm = 1.0/(np.sqrt(np.pi)*deltaz2)

        erf_min = sp.erf((zs-zmin)*edenoms)
        erf_max = sp.erf((zs-zmax)*edenoms)

        den = (-A*np.sqrt(np.pi)*(erf_max-erf_min)*zmax-np.sqrt(np.pi)*(A*erf_max-A*erf_min-erf_max+erf_min)*zmin+np.sqrt(np.pi)*zs*(2*A*erf_max-2*A*erf_min-erf_max+erf_min))/(np.sqrt(np.pi)*(deltaz2))+(-2*A*np.sqrt(2.0)*dzs+np.sqrt(2.0)*dzs)*np.exp(-0.5*np.square(zs - zmin)/dzs2)/(np.sqrt(np.pi)*(deltaz2))+(2*A*np.sqrt(2.0)*dzs-np.sqrt(2.0)*dzs)*np.exp(-0.5*np.square(zs-zmax)/dzs2)/(np.sqrt(np.pi)*(deltaz2))

        return np.log(den)
        
    def log_prior(self, p):
        r"""The log of the prior.  We use quasi-Jeffreys priors on all
        parameters:

        .. math::
        
          p(R) \sim \frac{1}{\sqrt{R}}

        .. math::

          p\left(\sigma_z\right) \sim \frac{\sigma_z}{\epsilon^2 + \sigma_z^2}

        where :math:`\epsilon` is the average observational errorbar.

        .. math::

          p\left(\mu_z\right) \sim \mathrm{const}

        .. math::

          p(A) \sim \frac{2A - 1 - \ln A + \ln (1-A)}{\left(1 - 2A\right)^3}

        """
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

        if muz - 2.0*sigmaz < self.zmin or muz + 2.0*sigmaz > self.zmax:
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
        """The log-likelihood of our model.

        """
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
        """Returns the sum of the log-likelihood and log-prior.

        """
        lp = self.log_prior(p)

        if lp == np.NINF:
            return np.NINF
        else:
            ll = self.log_likelihood(p)
            if np.isnan(ll):
                print p
                raise ValueError("ll = NaN")
            return lp + ll

    def log_foreground_probabilities(self, p):
        p = self.to_params(p)

        Rc = p['Rc']
        Rb = p['Rb']
        muz = p['muz']
        sigmaz = p['sigmaz']
        A = p['A']

        log_pfore = np.log(Rc) + self.log_convolved_foreground_density(muz, sigmaz)
        log_pback = np.log(Rb) + self.log_convolved_background_density(A)

        return log_pfore - np.logaddexp(log_pfore, log_pback)

class Posterior(object):
    def __init__(self, data, ncomp):
        self.data = data
        self.ncomp = ncomp
        self.ndim = data.shape[1]

    @property
    def nparams(self):
        ncomp = self.ncomp
        ndim = self.ndim

        return ncomp - 1 + ncomp*ndim + ncomp*(ndim+1)*ndim/2

    def to_params(self, p):
        ncomp = self.ncomp
        ndim = self.ndim
        
        p = np.atleast_1d(p)

        return p.view(np.dtype([('A', np.float, ncomp-1),
                                ('mu', np.float, (ncomp, ndim)),
                                ('cov', np.float, (ncomp, (ndim+1)*ndim/2))])).squeeze()

    def multivariate_gaussian_logpdf(self, mu, cov, xs):
        ys = xs - mu

        s, ldet = np.linalg.slogdet(cov)

        chi2 = np.sum(ys*np.linalg.solve(cov, ys.T).T, axis=1)

        return -0.5*xs.shape[1]*np.log(2.0*np.pi) - 0.5*ldet - 0.5*chi2

    def covariance_matrices(self, p):
        ndim = self.ndim
        ncomp = self.ncomp
        p = self.to_params(p)

        cms = np.zeros((ncomp, ndim, ndim))
        iu = np.triu_indices(ndim)

        for cm, cv in zip(cms, p['cov']):
            cm[iu] = cv
            cm += cm.T
            cm.reshape((-1,))[::(ndim+1)] /= 2.0

        return cms

    def params_from_amps_mus_covs(self, As, mus, covs):
        iu = np.triu_indices(self.ndim)
        sigmas = []
        for cov in covs:
            sigmas.append(cov[iu])
        sigmas = np.array(sigmas)

        return np.concatenate((As[:-1], mus.flatten(), sigmas.flatten()))

    def log_prior(self, p):
        p = self.to_params(p)

        cms = self.covariance_matrices(p)

        lp = 0.0

        if np.any(p['A'] < 0) or np.sum(p['A']) > 1:
            return np.NINF
        else:
            lp -= 0.5*(np.sum(np.log(p['A'])) + np.log1p(-np.sum(p['A'])))

        for mu in p['mu']:
            if np.any(mu < np.min(self.data, axis=0)) or np.any(mu > np.max(self.data, axis=0)):
                return np.NINF

        for cm in cms:
            lambdas = np.linalg.eigvalsh(cm)
            if np.any(lambdas < 0):
                return np.NINF
            else:
                lp -= 0.5*np.sum(np.log(lambdas))

        return lp

    def all_As(self, p):
        p = self.to_params(p)

        return np.concatenate((p['A'], [1.0-np.sum(p['A'])]))

    def log_likelihood(self, p):
        p = self.to_params(p)

        As = self.all_As(p)
        mus = p['mu']
        cms = self.covariance_matrices(p)

        logps = []
        for A, mu, cm in zip(As, mus, cms):
            logps.append(np.log(A) + self.multivariate_gaussian_logpdf(mu, cm, self.data))
        logps = np.array(logps)

        logps = np.logaddexp.reduce(logps, axis=0)

        return np.sum(logps)

    def __call__(self, p):
        lp = self.log_prior(p)

        if lp == np.NINF:
            return np.NINF
        else:
            return lp + self.log_likelihood(p)

    def log_ps(self, p):
        p = self.to_params(p)

        As = self.all_As(p)
        mus = p['mu']
        cms = self.covariance_matrices(p)

        logps = []
        for A, mu, cm in zip(As, mus, cms):
            logps.append(np.log(A) + self.multivariate_gaussian_logpdf(mu, cm, self.data))
        logps = np.array(logps)

        lognorms = np.logaddexp.reduce(logps, axis=0)

        return logps - lognorms
