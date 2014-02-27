def trim_data(data, zmin=0.0, zmax=0.1):
    r"""Trims the data given so that every data point is within 2-sigma of
    having :math:`z_\mathrm{min} \leq z \leq z_\mathrm{max}`.

    """

    zs = data[:,0]
    dzs = data[:,1]

    sel = (zs + 2.0*dzs >= zmin) & (zs - 2.0*dzs <= zmax)

    return data[sel, :]
