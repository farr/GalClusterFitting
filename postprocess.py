import numpy as np
import matplotlib.pyplot as plt

def pfirst_triangle_plot(logpost, chain):
    flatchain = chain.reshape((-1, chain.shape[2]))

    log_pfirst = np.NINF
    for p in flatchain:
        log_pfirst = np.logaddexp(log_pfirst, logpost.log_pfirst(p))
    log_pfirst -= np.log(flatchain.shape[0])

    labels = [r'$\alpha$', r'$\delta$', r'$cz$']

    for i in range(3):
        plt.subplot(3,3,4*i+1)
        plt.plot(logpost.data[:,i], np.exp(log_pfirst), '*k')
        plt.yscale('log')
        plt.xlabel(labels[i])
        plt.ylabel(r'$p({\rm cluster})$')

    for i in range(3):
        for j in range(i+1,3):
            plt.subplot(3,3,3*i+j+1)
            plt.scatter(logpost.data[:,i], logpost.data[:,j], c=log_pfirst)
            plt.xlabel(labels[i])
            plt.ylabel(labels[j])
