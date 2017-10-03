import sys
import numpy as np
import progressbar
import warnings
import time

from . import data
from . import grid
from .numpy_funcs import vector_gradient, curl, haloswap

class Sim(object):

    ndims = 3
    boundary = 1
    gamma0 = 2.211e5  # gyromagnetic ratio
    mu0 = 1.2566370614e-6 # permeability of free space

    def energy_expr(self):
        dx2 = 1 / self.grid_params.d**2
        dx = 1 / (2 * self.grid_params.d)
        dV = self.grid_params.prod_d
        e = np.array(self.sim_params.e)
        H = np.array(self.sim_params.H)
        Kc = dV * -self.sim_params.K
        Ac = dV * self.sim_params.A
        Dc = dV * -self.sim_params.D
        Hc = dV * -self.mu0 * self.sim_params.Ms

        return {"Zeeman":lambda m: Hc * np.tensordot(m, H, (0,0)),
            "Exchange":lambda m: Ac * vector_gradient(m, dx),
            "Anisotropy":lambda m: Kc * np.tensordot(m, e, (0,0)) ** 2,
            "DMI":lambda m: Dc * np.sum(m * curl(m, dx), axis=0)}

    def generate_energy_kernel(self):
        terms = []
        for key, value in self.energy_expr().items():
            if self.sim_params.has_term(key):
                terms.append(value)

        def energy(m):
            E = sum([f(m) for f in terms])
            return E

        return energy

    def generate_detailed_energy_kernel(self, terms):
        energy_expr = self.energy_expr()
        funcs = {k:energy_expr[k] for k in terms}

        def energy(d):
            ret = {}
            for key, value in funcs.items():
                E = []
                for dj in d:
                    E.append(value(dj))
                ret[key] = E

            return ret

        return energy

    def __init__(self, sim_params, grid_params, time_params, correction=0, periodic_boundary=True, save_every=1, print_params=False):
        self.__dict__.update({key:value for key, value in locals().items() if key not in ["self", "print_params"]})

        self.buffer_dims = tuple(int(n + 2 * self.boundary) for n in self.grid_params.n)
        self.buffer_slice = (slice(self.boundary,-self.boundary), ) * self.ndims
        self.data = data.Data(grid_params)

        self.need_generate_kernels = True
        self.initialised = False

        if print_params:
            print(self)
            print(self.data)
            sys.stdout.flush()

    def __str__(self):
        return "{}\nGRID\n{}\n\nTIME \n{}\n\tsave every = {}".format(self.sim_params, self.grid_params, self.time_params, self.save_every)

    def init(self, init_function=None, value=None):

        with warnings.catch_warnings():
            warnings.filterwarnings('error', category=RuntimeWarning)
            try:
                self.data.init()
                if not init_function is None:
                    if not value is None:
                        init_function(value)
                    else:
                        init_function()

                if self.need_generate_kernels:
                    print("Initialising simulation ...", file=sys.stderr)

                    start = time.time()
                    self.step_kernel = self.generate_step_kernel()
                    print("Generated step kernel: {:.2f} s".format(time.time() - start), file=sys.stderr)

                    start = time.time()
                    self.energy_kernel = self.generate_energy_kernel()
                    #print("Generated energy kernel: {:.2f} s".format(time.time() - start), file=sys.stderr)

                    start = time.time()
                    self.detailed_energy_kernel = self.generate_detailed_energy_kernel(self.sim_params.terms)
                    #print("Generated detailed energy kernel: {:.2f} s".format(time.time() - start), file=sys.stderr)

                    self.need_generate_kernels = False

            # Check for divide-by-zero (Warning)
            except (Warning) as e:
                print("Initialisation Error: ", type(e), e, file=sys.stderr)
                self.initialised = False
                return

        self.initialised = True

    def init_copy(self, value):
        self.init(self.data.set_copy, value)

    def init_random(self):
        self.init(self.data.set_random)

    def init_vector(self, value):
        self.init(self.data.set_vector, value)

    def init_disk(self, value=0.7):
        self.init(self.data.set_disk, value)

    def init_vortex(self):
        self.init(self.data.set_vortex)

    def init_flower(self):
        self.init(self.data.set_flower)

    def buffer_data(self, data):
        buffered_data = np.zeros((self.ndims,) + self.buffer_dims)
        buffered_data[(slice(None), ) + self.buffer_slice] = data
        if self.periodic_boundary:
            haloswap(buffered_data, *self.buffer_dims, self.boundary)
        return buffered_data

    def unbuffer_energy(self, data):
        return data[self.buffer_slice]

    def calc_energy(self, grid):
        return np.sum(self.unbuffer_energy(self.energy_kernel(self.buffer_data(grid[:]))))

    def calc_detailed_energy(self, data):
        buffered_data = []
        for grid in data:
            buffered_data.append(self.buffer_data(grid[:]))

        details = self.detailed_energy_kernel(buffered_data)

        ret = {}
        for name, es in details.items():
            if not name in ret:
                ret[name] = []
            for e in es:
                ret[name].append(np.sum(self.unbuffer_energy(e)))

        return ret

    def run_bounded(self):
        bar = progressbar.ProgressBar()
        steps = self.time_params.n // self.save_every
        for i in bar(range(steps)):
            self.data.push((i + 1) * self.time_params.d * self.save_every)
            self.step_kernel(self.data[-2].data, self.data[-1].data)

    def run_unbounded(self, energy_ratio_limit):
        if self.data.length > 1:
            last_energy, new_energy = self.calc_energy(self.data[-2].data), self.calc_energy(self.data[-1].data)
            energy_ratio = np.abs((new_energy - last_energy) / last_energy)
        else:
            last_energy, energy_ratio = self.calc_energy(self.data[-1].data), 1

        i = self.data.length
        bar = progressbar.ProgressBar(widgets=[progressbar.AnimatedMarker(), ' ', progressbar.Counter('%(value)05d'), ' ', progressbar.DynamicMessage('energy_ratio'), ' ', progressbar.Timer()], max_value=progressbar.UnknownLength)
        while energy_ratio > energy_ratio_limit:
            self.data.push((i + 1) * self.time_params.d * self.save_every)
            begin, end = self.data[-2].data, self.data[-1].data
            self.step_kernel(begin, end)
            new_energy = self.calc_energy(end)
            if new_energy > last_energy:
                raise Warning('Energy has increased')
            energy_ratio, last_energy = np.abs((new_energy - last_energy) / last_energy), new_energy
            i += 1
            bar.update(i * self.save_every, energy_ratio=energy_ratio)

    def run(self, energy_ratio_limit=1e-4):
        if self.initialised:
            print("Running simulation ...", file=sys.stderr)
            start = time.time()

            with warnings.catch_warnings():
                warnings.filterwarnings('error', category=RuntimeWarning)
                try:
                    if self.time_params.bounded:
                        self.run_bounded()
                    else:
                        self.run_unbounded(energy_ratio_limit)
                # Check for divide-by-zero (Warning)
                except (Warning) as e:
                    print("Simulation Error: ", type(e), e, file=sys.stderr)
                    return False, time.time() - start

            duration = time.time() - start
            print("Simulation Complete: {} steps, {} frames, {:.2f} s".format((self.data.length - 1) * self.save_every, self.data.length, duration), file=sys.stderr)
            sys.stderr.flush()
            return True, duration
        else:
            print("Skipping simulation: Not initialised", file=sys.stderr)
            return False, 0

    def plot(self, i, zs, image_width=10, filename=None, show_plot=True):
        self.data.plot(i, zs, image_width=image_width, filename=filename, show_plot=show_plot)

    def animate(self, zs, time_range=[None,None], max_frames=75, image_width=10, fps=15, filename=None):
        self.data.animate(zs, time_range=time_range, max_frames=max_frames, image_width=image_width, fps=fps, filename=filename)

    def plot_evolution(self, time_range=[None], test_names={"Norm", "Self-Alignment", "Zeeman Alignment", "Anisotropy Alignment"}, figsize=(7, 4), filename=None):
        tests = {}
        if "Norm" in test_names:
            tests["Norm"] = grid.norm
        if "Self-Alignment" in test_names:
            tests["Self-Alignment"] = grid.alignment
        if "Zeeman Alignment" in test_names and self.sim_params.has_term("zeeman"):
            tests["Zeeman Alignment"] = lambda m: grid.alignment(m, vector=self.sim_params.H)
        if "Anisotropy Alignment" in test_names and self.sim_params.has_term("anisotropy"):
            print("something")
            tests["Anisotropy Alignment"] = lambda m: grid.alignment(m, vector=self.sim_params.e, axial=True)

        time_slice = slice(*time_range)
        mean_results = {name:[] for name in tests}
        var_results = {name:[] for name in tests}
        for m in self.data[time_slice]:
            for name, test in tests.items():
                mean, var = m.apply_test(test, self.data.slices)
                mean_results[name].append(mean)
                var_results[name].append(var)

        self.data.plot_evolution(self.data.times[time_slice], mean_results, var_results, figsize=figsize, filename=filename)

    def plot_energy(self, time_range=[None], figsize=(7, 3), filename=None):
        time_slice = slice(*time_range)
        data_slice = self.data[time_slice]
        sum_energies = self.calc_detailed_energy(data_slice)
        normed_energies = {}
        total = np.zeros(len(data_slice))
        for name, data in sum_energies.items():
            data = np.array(data)
            total += data
            data_min, data_max = np.min(data), np.max(data)
            normed_energies[name] = (data - data_min) / (data_max - data_min)
        if len(sum_energies) > 1:
            sum_energies["total"] = total

        self.data.plot_energy(self.data.times[time_slice], sum_energies, normed_energies, figsize=figsize, filename=filename)

    def compare(self, other, figsize=(7, 3)):
        if self.data.length != other.data.length:
            print("Number of computed steps not equal ({} vs. {})".format(self.data.length, other.data.length))
            return False, {}
        elif self.data.shape != other.data.shape:
            print("Mesh shape not equal ({} vs. {})".format(self.data.shape, other.data.shape))
            return False, {}
        else:
            steps = np.arange(self.data.length) * self.save_every * self.time_params.d
            return True, self.data.plot_comparison(other.data, steps, "{} vs. {}".format(self.framework_name, other.framework_name), figsize)
