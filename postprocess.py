import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os.path as op
import plotutils.plotutils as pu
import triangle as tri

def _save_figure(path, name):
    if path is not None and name is not None:
        plt.savefig(op.join(path, name))

def triangle_plot(ftchain, logpost, path=None):
    tri.corner(ftchain, quantiles=[0.05, 0.95], labels=logpost.pnames)
    _save_figure(path, 'corner.pdf')

def plot_cluster_size(ftchain, logpost, path=None):
    sizes = []
    for p in ftchain:
        sizes.append(logpost.cluster_size(p))
    sizes = np.array(sizes)

    plt.figure()
    pu.plot_kde_posterior(sizes[:,0])
    plt.xlabel(r'$\sigma_\alpha$ (deg)')
    plt.ylabel(r'$p\left(\sigma_\alpha\right)$')
    _save_figure(path, 'ra-size.pdf')

    plt.figure()
    pu.plot_kde_posterior(sizes[:,1])
    plt.xlabel(r'$\sigma_\delta$ (deg)')
    plt.ylabel(r'$p\left(\sigma_\delta\right)$')
    _save_figure(path, 'dec-size.pdf')

    plt.figure()
    pu.plot_kde_posterior(sizes[:,2]*2.99792e5)
    plt.xlabel(r'$\sigma_z$ (km/s)')
    plt.ylabel(r'$p\left( \sigma_z \right)$')
    _save_figure(path, 'z-size.pdf')

def plot_rs(ftchain, logpost, path=None):
    rs = []
    for p in ftchain:
        rs.append(logpost.cluster_rs(p))
    rs = np.array(rs)

    plt.figure()
    pu.plot_kde_posterior(rs[:,0], label=r'$R_+$')
    pu.plot_kde_posterior(rs[:,1], label=r'$R_-$')
    plt.xlabel(r'$R$')
    plt.ylabel(r'$p(R)$')
    plt.legend()
    
    _save_figure(path, 'rs.pdf')

def plot_number(ftchain, logpost, path=None):
    n = logpost.zs.shape[0]

    Ns = np.random.gamma(n+0.5, size=ftchain.shape[0])

    Ncs = []
    for p, N in zip(ftchain, Ns):
        Ncs.append(p[0]*N)
    Ncs = np.array(Ncs)

    plt.figure()
    pu.plot_kde_posterior(Ncs)
    plt.xlabel(r'$N_\mathrm{cluster}$')
    plt.ylabel(r'$p\left(N_\mathrm{cluster}\right)$')
    _save_figure(path, 'N.pdf')

def plot_rho_cluster(ftchain, logpost, path=None):
    pftchain = logpost.to_params(ftchain)
    
    center = np.mean(pftchain['muc'][:,0:2], axis=0)
    widths = np.sqrt(np.mean(pftchain['covc'][:,[0, 3]], axis=0))

    xs = np.linspace(center[0]-5*widths[0], center[0] + 5*widths[0], 100)
    ys = np.linspace(center[1]-5*widths[1], center[1] + 5*widths[1], 100)

    XS, YS = np.meshgrid(xs, ys)

    pts = np.column_stack((XS.flatten(), YS.flatten()))

    log_ZS = np.NINF
    for p in ftchain:
        log_ZS = np.logaddexp(log_ZS, logpost.log_rho_sky(p, pts).reshape((100, 100)))
    log_ZS -= np.log(ftchain.shape[0])

    ZS = np.exp(log_ZS)
    ZS = 0.25*(ZS[:-1, :-1] + ZS[:-1, 1:] + ZS[1:, :-1] + ZS[1:, 1:])

    plt.figure()
    plt.pcolormesh(XS, YS, ZS, norm=mpl.colors.LogNorm())
    plt.colorbar()

    plt.plot(logpost.ras, logpost.decs, '.k')

    plt.axis(xmin=np.min(XS), xmax=np.max(XS), ymin=np.min(YS), ymax=np.max(YS))

    plt.xlabel(r'$\alpha$ (deg)')
    plt.ylabel(r'$\delta$ (deg)')
    plt.title(r'$dN_\mathrm{group}/dA$ (deg$^{-2}$)')

    _save_figure(path, 'rho-sky.pdf')

def plot_membership(ftchain, logpost, path=None):
    log_pfores = np.NINF
    for p in ftchain:
        log_pfores = np.logaddexp(log_pfores, logpost.log_pfore(p))
    log_pfores -= np.log(ftchain.shape[0])
    
    sel = (log_pfores > np.log(1e-8))

    plt.figure()
    plt.scatter(logpost.ras[sel], logpost.decs[sel], c=np.exp(log_pfores[sel]), s=50, norm=mpl.colors.LogNorm(), alpha=0.5)
    plt.xlabel(r'$\alpha$ (deg)')
    plt.ylabel(r'$\delta$ (deg)')
    plt.title(r'$p(\mathrm{member})$')
    plt.colorbar()

    _save_figure(path, 'pfores-sky.pdf')

def plot_all(ftchain, logpost, path=None):
    plot_cluster_size(ftchain, logpost, path=path)
    plot_rs(ftchain, logpost, path=path)
    plot_number(ftchain, logpost, path=path)
    plot_rho_cluster(ftchain, logpost, path=path)
    plot_membership(ftchain, logpost, path=path)
