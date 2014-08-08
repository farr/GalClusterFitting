#!/usr/bin/env python

"""Run a two-component fit to galaxy position and redshift data for a
cluster plus arbitrary background.  The fit requires two data files:

* A galaxy data file, with a header line containing at least ``ra``,
  ``dec``, ``z``, and ``dz``, and subsequent rows giving the measured
  position and redshift (and redshift uncertainty) of the galaxies in
  the sample.  The sample should extend some distance past the
  estimated boundaries of the cluster, at least to a few cluster radii
  and a few times the expected cluster velocity dispersion.  As long
  as the background is smooth, the fit will not be thrown off by
  including more background galaxies.

* A centroid data file, consisting of a header line containing at
  least ``ra0``, ``dec0``, ``z0``, and ``dz0``, and one row giving the
  estimated cluster centroid and velocity dispersion.

The output will consist of chain files and plots in the indicated
output directory.

"""

import argparse
import bz2
import emcee
import numpy as np
import os.path as op
import pickle
import plotutils.runner as pr
import postprocess as pp
import two_component as tc

def initialise_runner(args):
    galdata = np.genfromtxt(args.galaxies, names=True)
    centroid = np.genfromtxt(args.centroid, names=True)

    logpost = tc.TwoComponent(galdata['z'], galdata['dz'], galdata['ra'], galdata['dec'], 
                              centroid['z0'], centroid['dz0'], centroid['ra0'], centroid['dec0'])
    sampler = emcee.EnsembleSampler(args.nwalkers, logpost.nparams, logpost)
    return pr.EnsembleSamplerRunner(sampler, np.array([logpost.pguess() for i in range(args.nwalkers)]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--galaxies', metavar='FILE', required=True, help='galaxy data file')
    parser.add_argument('--centroid', metavar='FILE', required=True, help='centroid file')
    parser.add_argument('--outdir', metavar='DIR', default='.', help='output directory (default %(default)s)')

    parser.add_argument('--nwalkers', metavar='N', default=128, type=int, help='number of emcee walkers (default %(default)s)')
    parser.add_argument('--neff', metavar='N', default=128*64, type=int, help='number of effective samples (default %(default)s)')

    parser.add_argument('--restart', action='store_true', help='restart (if possible) an existing run')

    args = parser.parse_args()

    if args.restart:
        try:
          with bz2.BZ2File(op.join(args.outdir, 'runner.pkl.bz2'), 'r') as inp:
              runner = pickle.load(inp)
        except:
            runner = initialise_runner(args)
    else:
        runner = initialise_runner(args)

    runner.run_to_neff(args.neff / args.nwalkers, savedir=args.outdir)
    
    pp.plot_all(runner.thin_flatchain, runner.sampler.lnprobfn.f, path=args.outdir)
