import numpy as np
import scipy.special as sp
import scipy.stats as ss

def draw_data(Rc, Rb, muz, sigmaz, A, mu_noise = np.log(0.01), sigma_noise = 0.1, zmax=0.1):
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

    :param zmax: The maximum redshift

    """
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
    """Callable object representing the posterior.

    """
    
    def __init__(self, zs, dzs, zmax=0.1):
        """Initialize the posterior with the given redshifts and
        uncertainties.

        """
        self._zs = zs
        self._dzs = dzs
        self._zmax = zmax

    @property
    def zs(self):
        """The galaxy redshifts."""
        return self._zs

    @property
    def dzs(self):
        """Galaxy redshfit uncertainty."""
        return self._dzs

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
        zmax2 = zmax*zmax

        zs = self.zs
        dzs = self.dzs

        term1 = (A*zmax - 2.0*A*zs + zs)*(sp.erf((zmax - zs)/(np.sqrt(2.0)*dzs)) + sp.erf(zs/(np.sqrt(2.0)*dzs)))/zmax2

        term2 = np.sqrt(2.0)*dzs*(2.0*A-1.0)/(np.sqrt(np.pi)*zmax2)*(np.exp(-0.5*np.square(zmax-zs)/np.square(dzs)) - np.exp(-0.5*np.square(zs)/np.square(dzs)))

        return np.log(term1 + term2)
        
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
