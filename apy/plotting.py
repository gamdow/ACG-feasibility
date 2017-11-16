import numpy as np
import matplotlib.pyplot as plt

try:
    plt.plot(1,1)
    plt.close()
except:
    plt.switch_backend('agg')

dpi = 400

def frame(frame, zs, lengths=None, time=None, image_width=10, filename=None, show_plot=True):
    _, nx, ny, nz = frame.shape
    if lengths is not None:
        lx, ly, lz = lengths
    else:
        lx, ly, lz = nx, ny, nz

    if not hasattr(zs, "__iter__"):
        zs = [zs]
    num_z = len(zs)

    step = max(1, int(nx / (5 * image_width)))
    YS, XS = np.meshgrid(np.linspace(0, ly, ny), np.linspace(0, lx, nx))
    XS, YS = XS[::step, ::step], YS[::step, ::step]
    aspect_ratio = lx / ly
    scatter_scale = image_width * 2 * step * 10000 / (nx * ny)
    quiver_scale = 1.2 * np.sqrt(nx * ny) / step

    fig, ax = plt.subplots(num_z, 1, figsize=(image_width, num_z * image_width / aspect_ratio))
    P, Q, T = [], [], []
    if not hasattr(ax, "__iter__"):
        ax = [ax]
    for z, a in zip(zs, ax):
        mx, my, mz  = frame[:, ::step, ::step, z]
        a.ticklabel_format(style='sci', scilimits=(-2,2))
        if lengths is not None:
            a.set(xlabel=r'$x$ (m)', ylabel=r'$y$ (m)')
            time_string = r"$z = {:3.2e}$ m".format(z * lz / nz)
        else:
            a.set(xlabel=r'$x$', ylabel=r'$y$')
            time_string = r"$z = {}$".format(z)
        a.quiver(XS, YS, 0, 0, scale=1)
        P.append(a.scatter(XS, YS, s=mz ** 2 * scatter_scale, c=mz, cmap='bwr', vmin=-1, vmax=1))
        Q.append(a.quiver(XS, YS, mx, my, pivot='mid', scale=quiver_scale))
        if time is not None:
            time_string += r", $t={:3.2e}$ s".format(time)
        T.append(a.text(0, 0, time_string, fontsize=10,  bbox={'facecolor':'white', 'alpha':0.8, 'pad':5}))
        a.scatter(0, 0, s=scatter_scale, c="r", label=r"$Z+$")
        a.scatter(0, 0, s=scatter_scale, c="b", label=r"$Z-$")
        a.legend(loc='upper right')

    if not filename is None:
        plt.savefig(filename, bbox_inches='tight', dpi=dpi)

    if show_plot:
        plt.show()

    plt.close()

    return fig, step, Q, P, T, scatter_scale
