from matplotlib import colors as _colors
import colorspacious
import numpy as np
from scipy import interpolate, optimize

lab_to_rgb = colorspacious.cspace_converter('CAM02-UCS', 'sRGB1')
    
def uniform(colors=['black', '#f55', 'white'], N=256, lrange=(0, 100)):
    """
    Make a perceptually uniform colormap with increasing luminosity.
    
    Parameters
    ----------
    colors : sequence of matplotlib colors
        A sequence of colors. The hue of the colors will be preserved, but
        their luminosity will be changed and their positioning along the
        colormap scale will not be in general by evenly spaced steps.
    N : int
        The number of steps of the colormap. Default 256.
    lrange : sequence
        Two values for the start and end luminosity in range [0, 100].
        Default (0, 100).
    
    Return
    ------
    cmap : matplotlib.colors.ListedColormap
        A new colormap.
    
    Notes
    -----
    The colormap is uniform according to the CAM02-UCS colorspace, in the sense
    that the colorspace distance between the colors specified in the `colors`
    list, after changing their luminosity as needed, is proportional to their
    distance along the 0 to 1 scale of the colormap. The same holds for the
    substeps between these "nodes".
    
    The uniformity is preserved in grayscale, assuming that the conversion is
    done zeroing the chroma parameter of CIECAM02.
    
    Raises
    ------
    The function may fail if there are two consecutive similar colors in the
    list.
    """
    
    # TODO
    # Does not work when there are two consecutive similar colors. Tentative
    # solution: keep diff(l01) positive in the equations by mapping (-∞,∞)
    # to (0,∞) and then adding an equation that fixes the sum of the variables
    # to 1.
    
    rgb0 = np.array([_colors.to_rgb(color) for color in colors])
    lab0 = colorspacious.cspace_convert(rgb0, 'sRGB1', 'CAM02-UCS')
    
    lmin, lmax = lrange
    assert 0 <= lmin <= 100, lmin
    assert 0 <= lmax <= 100, lmax
    
    if len(lab0) > 2:
        l01 = computel01(lab0, lmin, lmax)
    else:
        l01 = np.array([0, 1])
    
    lab0[:, 0] = lmin + (lmax - lmin) * l01
    abinboundary(lab0)
    
    dist = np.sqrt(np.sum(np.diff(lab0, axis=0) ** 2, axis=1))
    distrel = dist / np.diff(l01)
    np.testing.assert_allclose(distrel, np.mean(distrel))
    
    kw = dict(axis=0, assume_sorted=True, copy=False)
    newx = np.linspace(0, 1, N)
    lab = interpolate.interp1d(l01, lab0, **kw)(newx)
    
    np.testing.assert_allclose(np.diff(lab[:, 0]), (lmax - lmin) / (N - 1))
    
    distsq = np.sum(np.diff(lab, axis=0) ** 2, axis=1)
    diff = np.diff(distsq)
    maxbad = 2 + 4 * (len(lab0) - 2)
    bad = np.count_nonzero(np.abs(diff) > 1e-8)
    assert bad <= maxbad, (bad, maxbad)

    rgb = lab_to_rgb(lab)
    rgb = np.clip(rgb, 0, 1)
    
    return _colors.ListedColormap(rgb)

def abinboundary(lab):
    # lab = array of triplets (l, a, b)
    # writes in-place
    
    def rgbok(l, a, b):
        rgb = lab_to_rgb([l, a, b])
        return np.max(np.abs(np.clip(rgb, 0, 1) - rgb)) < 1e-4

    def boundary(x, l, a, b):
        return rgbok(l, a * x, b * x) - 0.5

    for ilab in lab:
        l = ilab[0]
        if l < 0 or l > 100:
            ilab[1:] = 0
        elif not rgbok(*ilab):
            kw = dict(args=tuple(ilab), method='bisect', bracket=(0, 1))
            sol = optimize.root_scalar(boundary, **kw)
            assert sol.converged, sol.flag
            x = sol.root
            ilab[1:] *= x

def computel01(lab0, lmin, lmax):
    def diffsquares(x):
        # (x[i+1] - x[i])^2 - (x[i] - x[i-1])^2
        return (x[2:] - x[:-2]) * (x[:-2] - 2 * x[1:-1] + x[2:])

    def padl01(l01_var):
        return np.pad(l01_var, 1, constant_values=(0, 1))
    
    def equations(l01_var):
        lab = np.copy(lab0)
        l01 = padl01(l01_var)
        lab[:, 0] = lmin + (lmax - lmin) * l01
        abinboundary(lab)
        diffsq = np.diff(lab, axis=0) ** 2
        diffsqrel = diffsq / diffsq[:, :1]
        dist = np.sum(diffsqrel, axis=1)
        return np.diff(dist)

    l01_initial = np.linspace(0, 1, len(lab0))
    sol = optimize.root(equations, l01_initial[1:-1])
    assert sol.success, sol.message
    
    return padl01(sol.x)

def plotcmap(ax, cmap, N=512, **kw):
    img = np.linspace(0, 1, N)[None]
    return ax.imshow(img, cmap=cmap, aspect='auto', extent=(0, 1, 1, 0), **kw)

def plotab(J, abmax=50, N=512, **kw):
    Jab = np.empty((N, N, 3))
    Jab[..., 0] = J
    Jab[..., 1] = np.linspace(-abmax, abmax, N)
    Jab[..., 2] = Jab[..., 1].T
    rgb = colorspacious.cspace_convert(Jab, 'CAM02-UCS', 'sRGB1')
    bad = np.any((rgb < 0) | (rgb > 1), axis=-1, keepdims=True)
    rgb = np.where(bad, 1 if J < 50 else 0, rgb)

    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(num='colormap.plotab', clear=True)
    ax.set_xlabel('a')
    ax.set_ylabel('b')
    ax.set_title(f'J={J}')
    rt = ax.imshow(rgb, origin='lower', extent=(-abmax, abmax, -abmax, abmax), **kw)
    fig.show()
    return rt
