def trim_data(data, zmax=0.1):
    r"""Trims the data given so that every data point is within 2-sigma of
    having :math:`0 \leq z \leq z_\mathrm{max}`.

    """

    zs = data[:,0]
    dzs = data[:,1]

    sel = (zs + 2.0*dzs >= 0) & (zs - 2.0*dzs <= zmax)

    return data[sel, :]
