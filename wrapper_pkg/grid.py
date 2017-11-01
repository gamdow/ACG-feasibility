import numpy as np
from matplotlib import pyplot as plt

vector_index = 0
dpi = 400

def norm(val):
    return np.linalg.norm(val, axis=vector_index)

def normalise(val):
    return val / np.expand_dims(norm(val), vector_index)

def alignment(val, vector=None, axial=False):
    n = normalise(val)
    if vector is not None:
        # Alignment with given vector
        test_n = normalise(vector)
        dot = np.tensordot(np.array(test_n), n, (0, 0))
    else:
        # Self Alignment: Divide the data in two, equal randomly sampled sets and take inner product of the two
        data_list = n.reshape(3,-1).T
        np.random.seed(0)
        np.random.shuffle(data_list)
        np.random.seed(None)
        l = len(data_list)
        dot = np.sum(data_list[:l//2] * data_list[int(-(l-1)//2):], axis=1)
    if axial:
        dot = np.abs(dot)
    return dot

def expand_slice(space_slices=None):
    if space_slices is None:
        return (slice(None), )
    elif hasattr(space_slices, "__iter__"):
        return (slice(None), *space_slices)
    else:
        return (slice(None), space_slices)

class Grid(object):
    def __init__(self, space_dims):
        self.data = np.zeros((3, *space_dims))

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    @property
    def shape(self):
        return self.data.shape

    @property
    def size(self):
        return self.data.size

    @property
    def nbytes(self):
        return self.data.nbytes

    @property
    def dtype(self):
        return self.data.dtype

    def __str__(self):
        return "\nData Size: {:,} bytes {} {}".format(self.data.nbytes, self.data.shape, self.data.dtype)

    def copy(self, value, slices=slice(None)):
        self.data[slices] = value.data[slices]

    def set(self, func, slices=slice(None)):
        self.data[:] = 0
        data = self.data[slices]
        func(data)
        data[:] = normalise(data)

    def set_random(self, slices=slice(None)):
        def func(data):
            data[:] = np.random.rand(*np.shape(data)) * 2 - 1

        self.set(func, slices=slices)

    def set_vector(self, value, slices=slice(None)):
        def func(data):
            data[0] = value[0]
            data[1] = value[1]
            data[2] = value[2]

        self.set(func, slices=slices)

    def set_disk(self, value=0.7, slices=slice(None)):
        def func(data):
            center = np.array(np.shape(data)[1:]) / 2
            rx, ry = center[:2]
            r = min(rx, ry) * value
            data[0:2] = 0
            for index in np.ndindex(data[2].shape):
                d = r - np.linalg.norm(center[:2] - index[:2])
                data[(2,) + index] = 1 if d >= 0 else -1

        self.set(func, slices=slices)

    def set_vortex(self, slices=slice(None)):
        def func(data):
            center = np.array(np.shape(data)[1:]) / 2
            for index in np.ndindex(data[2].shape):
                x, y, z = center - index
                data[(slice(None),) + index] = y, -x, z

        self.set(func, slices=slices)

    def set_flower(self, slices=slice(None)):
        def func(data):
            center = np.array(np.shape(data)[1:]) / 2
            for index in np.ndindex(data[2].shape):
                if (center == index).all():
                    data[(slice(None),) + index] = [0,0,1]
                else:
                    data[(slice(None),) + index] = (center - index) * -1

        self.set(func, slices=slices)

    def apply_test(self, test, slices=slice(None)):
        r = test(self.data[slices])
        return np.average(r), np.var(r, ddof=1)

    def plot(self, zs, lengths=None, time=None, image_width=10, filename=None, show_plot=True):
        _, nx, ny, nz = self.data.shape
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
            mx, my, mz  = self.data[:, ::step, ::step, z]
            a.ticklabel_format(style='sci', scilimits=(-2,2))
            if lengths is not None:
                a.set(xlabel=r'$x$ (m)', ylabel=r'$y$ (m)')
                time_string = r"$z = {:3.2e}$ m".format(z * lz / nz)
            else:
                a.set(xlabel=r'$x$', ylabel=r'$y$')
                time_string = r"$z = {}$".format(z)
            a.quiver(XS, YS, 0, 0, scale=1)
            P.append(a.scatter(XS, YS, s=mz.reshape(-1) ** 2 * scatter_scale, c=mz.reshape(-1), cmap='bwr', vmin=-1, vmax=1))
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

        return fig, step, Q, P, T, scatter_scale
