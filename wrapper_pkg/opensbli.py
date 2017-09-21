import os
import sys
import numpy as np
from sympy import Matrix, Eq, solve
import progressbar
import shutil
import subprocess
import h5py

from opensbli.grid import Grid
from opensbli.spatial import Central, SpatialDiscretisation
from opensbli.timestepping import ForwardEuler, TemporalDiscretisation
from opensbli.problem import Problem
from opensbli.ics import GridBasedInitialisation
from opensbli.bcs import PeriodicBoundaryCondition
from opensbli.io import FileIO
from opensbli.opsc import OPSC

from . import sim

class Sim(sim.Sim):

    framework_name = "OpenSBLI"
    path = "opensbli_temp"

    def generate_step_kernel(self):
        if not self.periodic_boundary:
            raise NotImplementedError

        if not self.time_params.bounded:
            raise NotImplementedError

        if not self.correction == 0:
            raise NotImplementedError

        terms = {"Zeeman":"H_i",
            "Anisotropy":"Kc * m_j * e_j * e_i", # j
            "Exchange":"Ac * Der(Der(m_i, x_k), x_k)", # k
            "DMI":"Dc * LC(_i, _l, _m) * Der(m_m, x_l)"} # l, m

        heff_rhs = " + ".join([v for k, v in terms.items() if k in self.sim_params.terms])
        Heff = "Eq(Heff_i, {})".format(heff_rhs)
        LLForm = "Eq(Der(m_a, t), -gam / (1 + alp**2) * (LC(_a, _b, _i) * m_b * Heff_i + alp * LC(_a, _e, _d) * m_e * LC(_d, _c, _i) * m_c * Heff_i))"

        equations = [LLForm]
        substitutions = [Heff]
        constants = ["gam", "alp", "Ac", "Kc", "Dc", "e_i", "e_k", "e_j", "H_i"]
        coordinate_symbol = "x"
        metrics = [False, False, False]
        formulas = []
        problem = Problem(equations, substitutions, self.ndims, constants, coordinate_symbol, metrics, formulas)
        expanded_formulas = problem.get_expanded(problem.formulas)
        expanded_equations = problem.get_expanded(problem.equations)

        spatial_order = 2
        spatial_scheme = Central(spatial_order)
        grid = Grid(3, {'delta':tuple(float(i) for i in self.grid_params.d), 'number_of_points':tuple(int(i) for i in self.grid_params.n)})
        spatial_discretisation = SpatialDiscretisation(expanded_equations, expanded_formulas, grid, spatial_scheme)

        temporal_scheme = ForwardEuler()
        constant_dt = True
        temporal_discretisation = TemporalDiscretisation(temporal_scheme, grid, constant_dt, spatial_discretisation)

        boundary_condition = PeriodicBoundaryCondition(grid)
        for dim in range(self.ndims):
            # Apply the boundary condition in all directions.
            boundary_condition.apply(arrays=temporal_discretisation.prognostic_variables, boundary_direction=dim)

        initial_conditions = GridBasedInitialisation(grid, self.initial_conditions)

        io = FileIO(temporal_discretisation.prognostic_variables, self.save_every)

        c = 2 / (self.mu0 * self.sim_params.Ms)
        steps = (self.time_params.n // self.save_every if self.time_params.bounded else 1) * self.save_every

        simulation_parameters = {'name': "LLG",
                                 'precision': "double",
                                 'deltat': self.time_params.d,
                                 'niter': steps - 1,
                                 'gam': self.gamma0,
                                 'alp': self.sim_params.alpha,
                                 'Ac': self.sim_params.A * c,
                                 'Kc': self.sim_params.K * c,
                                 'Dc': self.sim_params.D * c,
                                 'e0': self.sim_params.e[0],
                                 'e1': self.sim_params.e[1],
                                 'e2': self.sim_params.e[2],
                                 'H0': self.sim_params.H[0],
                                 'H1': self.sim_params.H[1],
                                 'H2': self.sim_params.H[2]}

        red_eq = []

        shutil.rmtree(self.path, ignore_errors=True)
        os.mkdir(self.path)

        OPSC(grid, spatial_discretisation, temporal_discretisation, boundary_condition, initial_conditions, io, simulation_parameters, red_eq)

        os.rename("LLG_opsc_code", "{}/LLG_opsc_code".format(self.path))

        module_dir = os.path.dirname(os.path.realpath(__file__))
        shutil.copy2(module_dir+"/OpenSBLI_LLG_Makefile","{}/LLG_opsc_code/Makefile".format(self.path))

        out = subprocess.Popen(["make", "LLG_openmp"],  cwd="{}/LLG_opsc_code".format(self.path))
        stdout, stderr = out.communicate()
        if out.returncode:
            if not stdout is None:
                print(stdout.decode('utf-8'), file=sys.stderr)
            if not stderr is None:
                print(stderr.decode('utf-8'), file=sys.stderr)
            sys.stderr.flush()
            raise RuntimeError

        def step(f, t):
            pass

        return step

    def run_bounded(self):
        out = subprocess.Popen("LLG_opsc_code/LLG_openmp", cwd="{}".format(self.path), stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        stdout, stderr = out.communicate()
        if out.returncode:
            if not stdout is None:
                print(stdout.decode('utf-8'), file=sys.stderr)
            if not stderr is None:
                print(stderr.decode('utf-8'), file=sys.stderr)
            sys.stderr.flush()
            raise RuntimeError

        h5_files = [f for f in os.listdir(self.path) if (os.path.isfile(os.path.join(self.path, f)) and f.startswith("LLG") and f.endswith(".h5"))]
        key=lambda x: int(x[4:-3])
        h5_files.sort(key=key)
        for f in h5_files:
            n = key(f)
            self.data.push((n + 1) * self.time_params.d)
            path = os.path.join(self.path, f)
            with h5py.File(path,'r') as h5f:
                grid = self.data[-1]
                for j in range(3):
                    grid[j] = np.transpose(h5f['LLG_block']['m' + str(j)])[self.buffer_slice]
            #os.remove(path)

    def run_unbounded(self, energy_ratio_limit):
        raise NotImplementedError
        pass


    def init_copy(self, value):
        raise NotImplementedError
        pass

    def init_random(self):
        raise NotImplementedError
        pass

    def init_vector(self, value):
        self.initial_conditions = [
                      "Eq(grid.work_array(m0), {}".format(value[0]),
                      "Eq(grid.work_array(m1), {}".format(value[1]),
                      "Eq(grid.work_array(m2), {}".format(value[2])]
        self.need_generate_kernels = True
        super(Sim, self).init_vector(value)

    def init_disk(self, value=0.7):
        center = self.grid_params.n / 2
        rx, ry = center[:2]
        r = min(rx, ry) * value

        radius = "Eq(grid.grid_variable(radius), {r} - pow(pow({rx} - grid.Idx[0], 2) + pow({ry} - grid.Idx[1], 2), 0.5))".format(r=r, rx=rx, ry=ry)
        self.initial_conditions = [radius,
                      "Eq(grid.work_array(m0), 0)",
                      "Eq(grid.work_array(m1), 0)",
                      "Eq(grid.work_array(m2), sign(radius))"]
        self.need_generate_kernels = True
        super(Sim, self).init_disk(value)

    def init_vortex(self):
        cx, cy, cz = self.grid_params.n / 2

        rx = "Eq(grid.grid_variable(rx), {} - grid.Idx[0])".format(cx)
        ry = "Eq(grid.grid_variable(ry), {} - grid.Idx[1])".format(cy)
        rz = "Eq(grid.grid_variable(rz), {} - grid.Idx[2])".format(cz)
        norm = "Eq(grid.grid_variable(norm), pow(pow(rx,2) + pow(ry,2) + pow(rz,2),0.5))".format(cz)
        self.initial_conditions = [rx, ry, rz, norm,
                      "Eq(grid.work_array(m0), ry / norm)",
                      "Eq(grid.work_array(m1), -rx / norm)",
                      "Eq(grid.work_array(m2), rz / norm)"]
        self.need_generate_kernels = True
        super(Sim, self).init_vortex()


    def init_flower(self):
        cx, cy, cz = self.grid_params.n / 2

        rx = "Eq(grid.grid_variable(rx), {} - grid.Idx[0])".format(cx)
        ry = "Eq(grid.grid_variable(ry), {} - grid.Idx[1])".format(cy)
        rz = "Eq(grid.grid_variable(rz), {} - grid.Idx[2])".format(cz)
        norm = "Eq(grid.grid_variable(norm), pow(pow(rx,2) + pow(ry,2) + pow(rz,2),0.5))".format(cz)
        self.initial_conditions = [rx, ry, rz, norm,
                      "Eq(grid.work_array(m0), -rx / norm)",
                      "Eq(grid.work_array(m1), -ry / norm)",
                      "Eq(grid.work_array(m2), -rz / norm)"]
        self.need_generate_kernels = True
        super(Sim, self).init_flower()
