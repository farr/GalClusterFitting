from emcee import EnsembleSampler
import glob
from gzip import GzipFile
import matplotlib.pyplot as pp
import numpy as np
import os.path as op
import posterior as pos
import scipy.stats as ss
import triangle as tri
import warnings

def true_density(p, z, zmin=0.0, zmax=0.1):
    lc, lb, mu, sigma, A = p

    dz = zmax-zmin
    dz2 = dz*dz

    return (lc*ss.norm.pdf(z, loc=mu, scale=sigma) + 2.0*lb*(A*(zmax-z)/dz2 + (1.0-A)*(z-zmin)/dz2))/(lc+lb)

def corner_plot(chain, outdir=None):
    tri.corner(chain.reshape((-1, chain.shape[2])), labels=pos.parameter_labels, quantiles=[0.05, 0.95])
    if outdir is not None:
        pp.savefig(op.join(outdir, 'corner.pdf'))

def data_model_plot(data, sampler, outdir=None, zmin=0.0, zmax=0.1, nzs=1000):
    pp.figure()

    dz = np.mean(sampler.chain[:,:,3])
    bins = 5*int(round((zmax-zmin)/dz))

    pp.hist(data[:,0], bins, normed=True, histtype='step')

    zs = np.linspace(zmin, zmax, nzs)
    pzs = np.zeros(nzs)

    for p in sampler.flatchain:
        pzs += true_density(p, zs, zmin=zmin, zmax=zmax)
    pzs /= sampler.flatchain.shape[0]
    pp.plot(zs, pzs, '-k')

    pp.xlabel(r'$z$')
    pp.ylabel(r'$1/N dN_\mathrm{total}/dz$')

    pp.axis(xmin=zmin, xmax=zmax)

    if outdir is not None:
        pp.savefig(op.join(outdir, 'dNdz.pdf'))

def pfore_plot(logpost, sampler, outdir=None, N=10000):
    pp.figure()

    iskip = max(1, int(np.floor(sampler.flatchain.shape[0]/float(N))))

    thin_chain = sampler.flatchain[::iskip,:]
    
    log_ps = np.zeros(logpost.zs.shape[0])
    log_ps[:] = np.NINF
    for p in thin_chain:
        log_ps = np.logaddexp(log_ps, logpost.log_foreground_probabilities(p))
    log_ps -= np.log(thin_chain.shape[0])

    inds = np.argsort(logpost.zs)

    pp.errorbar(logpost.zs[inds], np.exp(log_ps)[inds], fmt='.', color='k', xerr=logpost.dzs)

    pp.xlabel(r'$z_\mathrm{obs}$')
    pp.ylabel(r'$p(\mathrm{member})$')

    if outdir is not None:
        pp.savefig(op.join(outdir, 'pmember.pdf'))

def postprocess(outdir, data, logpost, sampler, zmin=0.0, zmax=0.1):
    corner_plot(sampler.chain, outdir=outdir)
    data_model_plot(data, sampler, outdir=outdir, zmin=zmin, zmax=zmax)
    pfore_plot(logpost, sampler, outdir=outdir)

    with GzipFile(op.join(outdir, 'chain.npy.gz'), 'w') as out:
        np.save(out, sampler.chain)
    with GzipFile(op.join(outdir, 'lnprob.npy.gz'), 'w') as out:
        np.save(out, sampler.lnprobability)

def trim_data(data, zmin=0.0, zmax=0.1):
    r"""Trims the data given so that every data point is within 2-sigma of
    having :math:`z_\mathrm{min} \leq z \leq z_\mathrm{max}`.

    """

    zs = data[:,0]
    dzs = data[:,1]

    sel = (zs + 2.0*dzs >= zmin) & (zs - 2.0*dzs <= zmax)

    return data[sel, :]

def load_data_logpost_sampler(outdir, zmin=0.0, zmax=0.1):
    files = glob.glob(op.join(outdir, 'MZ*'))
    if len(files) > 1:
        warnings.warn('more than one input file found; choosing the first')
    data = np.loadtxt(files[0])

    data = trim_data(data, zmin=zmin, zmax=zmax)
    logpost = pos.Posterior(data[:,0], data[:,1], zmin=zmin, zmax=zmax)
    sampler = EnsembleSampler(100, 5, logpost)

    with GzipFile(op.join(outdir, 'chain.npy.gz'), 'r') as inp:
        sampler._chain = np.load(inp)
    with GzipFile(op.join(outdir, 'lnprob.npy.gz'), 'r') as inp:
        sampler._lnprob = np.load(inp)

    return data, logpost, sampler

