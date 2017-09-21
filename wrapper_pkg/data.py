import sys
import numpy as np
from matplotlib import pyplot as plt, animation, rc
from matplotlib.ticker import LogLocator
from IPython.display import HTML, display
import progressbar

from .grid import Grid, dpi, expand_slice

rc('animation', html='html5')

def plot(xs, ys, set_labels=None, figsize=(7, 3)):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig.tight_layout()
    if set_labels is not None:
        ax.set(**set_labels)
    if type(ys) is dict:
        for name, data in ys.items():
            ax.plot(xs, data, label=name)
    else:
        ax.plot(xs, ys)
    ax.ticklabel_format(style='sci', scilimits=(-2,2), axis='y')
    ax.legend()
    return ax

class Data(object):
    def __init__(self, params, slices=slice(None)):
        self.dims = params.n
        self.lengths = params.l
        self.slices = slices
        self.init()

    def __getitem__(self, key):
        return self.datas[key]

    def __setitem__(self, key, value):
        self.datas[key] = value

    @property
    def length(self):
        return len(self.datas)

    @property
    def shape(self):
        return (self.length,) + self.datas[0].shape

    @property
    def slice(self):
        return expand_slice(self.slices)

    def __str__(self):
        nbytes = sum(grid.nbytes for grid in self.datas)
        return "\nData Size: {:,} bytes {} {}".format(nbytes, self.shape, self.datas[0].dtype)

    def init(self):
        self.times = [0]
        self.datas = [Grid(self.dims)]

    def set_copy(self, value):
        self.datas[0].copy(value, self.slices)

    def set_random(self):
        self.datas[0].set_random(self.slices)

    def set_vector(self, value):
        self.datas[0].set_vector(value, self.slices)

    def set_disk(self, value=0.7):
        self.datas[0].set_disk(value, self.slices)

    def set_vortex(self):
        self.datas[0].set_vortex(self.slices)

    def set_flower(self):
        self.datas[0].set_flower(self.slices)

    def push(self, time):
        self.times.append(time)
        self.datas.append(Grid(self.dims))

    def plot(self, i, zs, image_width=7, filename=None, show_plot=True):
        return self.datas[i].plot(zs, self.lengths, self.times[i], image_width=image_width, filename=filename, show_plot=show_plot)

    def animate(self, zs, time_range=[None], max_frames=75, image_width=7, fps=15):
        print("Rendering animation ...", file=sys.stderr)
        fig, space_step, Q, P, T, scatter_scale = self.plot(0, zs, image_width=image_width, show_plot=False)

        if not hasattr(zs, "__iter__"):
            zs = [zs]

        time_slice = slice(*time_range)
        data = self.datas[time_slice]
        time_step = (len(data) + max_frames - 1 ) // max_frames if max_frames > 0 else 1
        data = data[::time_step]
        times = self.times[time_slice][::time_step]

        steps = len(data)
        with progressbar.ProgressBar(max_value=steps) as bar:
            def do_steps(i):
                for j, z in enumerate(zs):
                    mx, my, mz  = data[i][:,::space_step,::space_step,z]
                    Q[j].set_UVC(mx, my)
                    P[j].set_array(mz.reshape(-1))
                    P[j].set_sizes(mz.reshape(-1) ** 2 * scatter_scale)
                    T[j].set_text(r"$z = {:3.2e}$ m, $t={:3.2e} $s".format(z * self.lengths[2] / self.dims[2], times[i]))
                    bar.update(i + 1)
                return Q,

            anim = animation.FuncAnimation(fig, do_steps, range(steps), interval=1000/fps)
            ret = HTML(anim.to_html5_video())

        plt.close()

        display(ret)

    def plot_evolution(self, times, mean_results, var_results, figsize, filename=None):
        figsize = (figsize[0], figsize[1] / 2)
        plot(times, mean_results, set_labels={"xlabel":r'Time (s)', "ylabel":"Mean"}, figsize=figsize)
        if not filename is None:
            plt.savefig(filename.replace('.png', '_mean.png'), bbox_inches='tight', dpi=dpi)
        plot(times, var_results, set_labels={"xlabel":r'Time (s)', "ylabel":"Variance"}, figsize=figsize)
        if not filename is None:
            plt.savefig(filename.replace('.png', '_variance.png'), bbox_inches='tight', dpi=dpi)
        plt.show()

    def plot_energy(self, times, sum_energies, normed_energies, figsize, filename=None):
        ax = plot(times, sum_energies, {"xlabel":r"Time (s)", "ylabel":r"Energy (J)"}, figsize=figsize)
        if not filename is None:
            plt.savefig(filename, bbox_inches='tight', dpi=dpi)
        plt.show()

    def plot_comparison(self, other, steps, title, figsize):
        def norm(data):
            return np.sum(data * data, axis=0)

        norm_res = []
        inner = []
        for g1, g2 in zip(self.datas, other.datas):
            norm_res.append(np.average(np.abs(norm(g1.data) - norm(g2.data))))
            inner.append(np.average(1 - np.sum(g1.data * g2.data, axis=0)))

        results = {r"Norm Residual":norm_res, "Inner Product Residual":inner}
        ax = plot(steps, results, {"xlabel":r"Time (s)", "ylabel":r"", "title":title}, figsize=figsize)
        plt.show()
        return results
