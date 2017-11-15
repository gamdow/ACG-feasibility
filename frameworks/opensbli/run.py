from __future__ import print_function
import os
import sys

if len(sys.argv) < 3:
    print("*** ERROR: Missing input and output filenames ***", file=sys.stderr)
    sys.exit()
input_file = str(sys.argv[1])
output_file = str(sys.argv[2])

import numpy as np
from sympy import Matrix, Eq, solve
import subprocess
import shutil
import h5py
import glob
import apy

from opensbli.grid import Grid
from opensbli.spatial import Central, SpatialDiscretisation
from opensbli.timestepping import ForwardEuler, RungeKutta,  TemporalDiscretisation
from opensbli.problem import Problem
from opensbli.ics import GridBasedInitialisation
from opensbli.bcs import PeriodicBoundaryCondition
from opensbli.io import FileIO
from opensbli.opsc import OPSC
import opensbli
import logging
opensbli.LOG.setLevel(logging.ERROR)

settings = apy.Struct(apy.read_settings(input_file))
buffer_params = apy.Struct(apy.buffer_params(settings, 1))

if not settings.periodic_boundary:
    raise NotImplementedError

if not settings.correction == 0:
    raise NotImplementedError

terms = {"Zeeman":"H_i",
    "Anisotropy":"Kc * m_j * e_j * e_i", # j
    "Exchange":"Ac * Der(Der(m_i, x_k), x_k)", # k
    "DMI":"Dc * LC(_i, _l, _m) * Der(m_m, x_l)"} # l, m

heff_rhs = " + ".join([v for k, v in terms.items() if apy.has_term(settings, k)])
Heff = "Eq(Heff_i, {})".format(heff_rhs)
LLForm = "Eq(Der(m_a, t), -gam / (1 + alp**2) * (LC(_a, _b, _i) * m_b * Heff_i + alp * LC(_a, _e, _d) * m_e * LC(_d, _c, _i) * m_c * Heff_i))"

equations = [LLForm]
substitutions = [Heff]
constants = ["gam", "alp", "Ac", "Kc", "Dc", "e_i", "e_k", "e_j", "H_i"]
coordinate_symbol = "x"
metrics = [False, False, False]
formulas = []
problem = Problem(equations, substitutions, len(settings.grid.n), constants, coordinate_symbol, metrics, formulas)
expanded_formulas = problem.get_expanded(problem.formulas)
expanded_equations = problem.get_expanded(problem.equations)

spatial_order = 2
spatial_scheme = Central(spatial_order)
grid = Grid(3, {'delta':tuple(float(i) for i in settings.grid.d), 'number_of_points':tuple(int(i) for i in settings.grid.n)})
spatial_discretisation = SpatialDiscretisation(expanded_equations, expanded_formulas, grid, spatial_scheme)

constant_dt = True
temporal_discretisation = TemporalDiscretisation(RungeKutta(3), grid, constant_dt, spatial_discretisation)

boundary_condition = PeriodicBoundaryCondition(grid)
for dim in range(len(settings.grid.n)):
    # Apply the boundary condition in all directions.
    boundary_condition.apply(arrays=temporal_discretisation.prognostic_variables, boundary_direction=dim)

def init_vector(value):
    return [
      "Eq(grid.work_array(m0), {}".format(value[0]),
      "Eq(grid.work_array(m1), {}".format(value[1]),
      "Eq(grid.work_array(m2), {}".format(value[2])]

def init_disk(value=0.7):
    center = np.array(settings.grid.n) / 2
    rx, ry = center[:2]
    r = min(rx, ry) * value
    radius = "Eq(grid.grid_variable(radius), {r} - pow(pow({rx} - grid.Idx[0], 2) + pow({ry} - grid.Idx[1], 2), 0.5))".format(r=r, rx=rx, ry=ry)
    return [radius,
      "Eq(grid.work_array(m0), 0)",
      "Eq(grid.work_array(m1), 0)",
      "Eq(grid.work_array(m2), sign(radius))"]

def init_vortex(value=None):
    cx, cy, cz = np.array(settings.grid.n) / 2
    rx = "Eq(grid.grid_variable(rx), {} - grid.Idx[0])".format(cx)
    ry = "Eq(grid.grid_variable(ry), {} - grid.Idx[1])".format(cy)
    rz = "Eq(grid.grid_variable(rz), {} - grid.Idx[2])".format(cz)
    norm = "Eq(grid.grid_variable(norm), pow(pow(rx,2) + pow(ry,2) + pow(rz,2),0.5))".format(cz)
    return [rx, ry, rz, norm,
      "Eq(grid.work_array(m0), Piecewise((ry / norm, norm > 0), (0, True)))",
      "Eq(grid.work_array(m1), Piecewise((-rx / norm, norm > 0), (0, True)))",
      "Eq(grid.work_array(m2), Piecewise((rz / norm, norm > 0), (0, True)))"]

def init_flower(value=None):
    cx, cy, cz = np.array(settings.grid.n) / 2
    rx = "Eq(grid.grid_variable(rx), {} - grid.Idx[0])".format(cx)
    ry = "Eq(grid.grid_variable(ry), {} - grid.Idx[1])".format(cy)
    rz = "Eq(grid.grid_variable(rz), {} - grid.Idx[2])".format(cz)
    norm = "Eq(grid.grid_variable(norm), pow(pow(rx,2) + pow(ry,2) + pow(rz,2),0.5))".format(cz)
    return [rx, ry, rz, norm,
      "Eq(grid.work_array(m0), Piecewise((-rx / norm, norm > 0), (0, True)) )",
      "Eq(grid.work_array(m1), Piecewise((-ry / norm, norm > 0), (0, True)) )",
      "Eq(grid.work_array(m2), Piecewise((-rz / norm, norm > 0), (1, True)) )"]

init_functions = {"vector":init_vector,
    "disk":init_disk,
    "vortex":init_vortex,
    "flower":init_flower}

if not settings.init in init_functions:
    raise NotImplementedError

initial_conditions = GridBasedInitialisation(grid, init_functions[settings.init](settings.value))

io = FileIO(temporal_discretisation.prognostic_variables, settings.save_every)

c = 2 / (settings.mu0 * settings.Ms)
steps = (settings.time.n // settings.save_every) * settings.save_every

simulation_parameters = {'name': "LLG",
                         'precision': "double",
                         'deltat': settings.time.d,
                         'niter': steps - 1,
                         'gam': settings.gamma0,
                         'alp': settings.alpha,
                         'Ac': settings.A * c,
                         'Kc': settings.K * c,
                         'Dc': settings.D * c,
                         'e0': settings.e[0],
                         'e1': settings.e[1],
                         'e2': settings.e[2],
                         'H0': settings.H[0],
                         'H1': settings.H[1],
                         'H2': settings.H[2]}

red_eq = []

print("Generating code ...", file=sys.stderr)
OPSC(grid, spatial_discretisation, temporal_discretisation, boundary_condition, initial_conditions, io, simulation_parameters, red_eq)

shutil.copyfile("Makefile", "LLG_opsc_code/Makefile")

print("Compiling code ...", file=sys.stderr)
out = subprocess.Popen(["make", "LLG_openmp"], cwd="LLG_opsc_code", stderr=subprocess.PIPE, stdout=subprocess.PIPE)
stdout, stderr = out.communicate()
if out.returncode:
    if not stdout is None:
        print(stdout.decode('utf-8'), file=sys.stderr)
    if not stderr is None:
        print(stderr.decode('utf-8'), file=sys.stderr)
    sys.stderr.flush()
    raise RuntimeError

num_threads = int(os.environ['OMP_NUM_THREADS']) if 'OMP_NUM_THREADS' in os.environ else 1
data = apy.Data(settings, "OpenSBLI", num_threads)

print("Running simulation ...", file=sys.stderr)
data.start_timer()
out = subprocess.Popen("LLG_opsc_code/LLG_openmp", stderr=subprocess.PIPE, stdout=subprocess.PIPE)
stdout, stderr = out.communicate()
if out.returncode:
    if not stdout is None:
        print(stdout.decode('utf-8'), file=sys.stderr)
    if not stderr is None:
        print(stderr.decode('utf-8'), file=sys.stderr)
    sys.stderr.flush()
    raise RuntimeError

key=lambda x: int(x[4:-3])
for f in sorted(glob.glob("LLG*.h5"), key=key):
    n = key(f)
    data.push((n + 1) * settings.time.d)
    with h5py.File(f, 'r') as h5f:
        grid = data[-1]
        for j in range(3):
            grid[j] = np.transpose(h5f['LLG_block']['m' + str(j)])[buffer_params.buffer_slice]
data.end_timer()

data.dump(output_file)
