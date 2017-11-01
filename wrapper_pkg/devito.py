import os
import sys

if not "DEVITO_OPENMP" in os.environ or os.environ["DEVITO_OPENMP"] != "1":
    print("*** WARNING: Devito OpenMP environment variable has not been set ***", file=sys.stderr)


import numpy as np
from sympy import Matrix, Eq, solve
import progressbar
from devito import TimeData, Operator, t, x, y, z, logger as devito_logger, parameters as devito_parameters

from . import sim

devito_logger.set_log_level('WARNING')

def vector_laplacian(u):
    return Matrix([u[0].dx2 + u[0].dy2 + u[0].dz2,
                   u[1].dx2 + u[1].dy2 + u[1].dz2,
                   u[2].dx2 + u[2].dy2 + u[2].dz2])

def vector_gradient(u):
    return u[0].dx**2 + u[0].dy**2 + u[0].dz**2 + u[1].dx**2 + u[1].dy**2 + u[1].dz**2 + u[2].dx**2 + u[2].dy**2 + u[2].dz**2

def curl(u):
    return Matrix([u[2].dy - u[1].dz,
                   u[0].dz - u[2].dx,
                   u[1].dx - u[0].dy])

expression_cache = {}

class Sim(sim.Sim):

    framework_name = "Devito"

    @property
    def data_shape(self):
        # Devito doesn't like numpy types for the grid dimensions, and it needs to be a tuple, so shape needs to be converted
        return tuple(int(i) for i in self.grid_params.n)

    def data_matrix(self, settings):
        return Matrix([TimeData(name='m_x', **settings),
            TimeData(name='m_y', **settings),
            TimeData(name='m_z', **settings)])

    def generate_step_kernel(self):
        settings = {"shape":self.buffer_dims, "space_order":2}
        m = self.data_matrix(settings)
        c = 2 / (self.mu0 * self.sim_params.Ms)
        zeeman = Matrix(self.sim_params.H)
        exchange = self.sim_params.A * c * vector_laplacian(m)
        e = Matrix(self.sim_params.e)
        anisotropy = self.sim_params.K * c * m.dot(e) * e
        dmi = self.sim_params.D * c * curl(m)
        heff = zeeman + exchange + anisotropy + dmi
        crossHeff = m.cross(heff)
        dmdt_rhs = -self.gamma0 / (1 + self.sim_params.alpha**2) * (crossHeff + self.sim_params.alpha * m.cross(crossHeff))
        dmdt_lhs = Matrix([TimeData(name='dmdt_x', **settings),
            TimeData(name='dmdt_y', **settings),
            TimeData(name='dmdt_z', **settings)])
        dmdt_correction = self.correction * dmdt_lhs.dot(dmdt_lhs)**0.5 * (1 - m.dot(m)) * m
        string_llg = str(dmdt_rhs) + str(dmdt_correction)

        if string_llg in expression_cache:
            update = expression_cache[string_llg]
        else:
            update = []
            if self.correction > 0:
                # if using correction solve in 2 steps; calculate dmdt, then calculate m[t+1] = dmdt + correction
                for i, dmdti in enumerate(dmdt_lhs):
                    update.append(Eq(dmdti, dmdt_rhs[i]))
                llg_eqn = Matrix([mi.dt for mi in m]) - (dmdt_lhs + dmdt_correction)
            else:
                # if not using correction; m[t+1] = dmdt
                llg_eqn = Matrix([mi.dt for mi in m]) - dmdt_rhs

            print("Solving LLG Sympy expressions ...", file=sys.stderr)
            with progressbar.ProgressBar(max_value=len(m)) as bar:
                for i, mi in enumerate(m):
                    update.append(Eq(mi.forward, solve(llg_eqn[i], mi.forward)[0]))
                    bar.update(i + 1)
            expression_cache[string_llg] = update

        bcs = []
        nx, ny, nz = self.buffer_dims
        if self.periodic_boundary:
            for mi in m:
                bcs += [Eq(mi.indexed[t, x, y, 0], mi.indexed[t, x, y, nz - 2])]
                bcs += [Eq(mi.indexed[t, x, y, nz - 1], mi.indexed[t, x, y, 1])]
                bcs += [Eq(mi.indexed[t, x, 0, z], mi.indexed[t, x, ny - 2, z])]
                bcs += [Eq(mi.indexed[t, x, ny - 1, z], mi.indexed[t, x, 1, z])]
                bcs += [Eq(mi.indexed[t, 0, y, z], mi.indexed[t, nx - 2, y, z])]
                bcs += [Eq(mi.indexed[t, nx - 1, y, z], mi.indexed[t, 1, y, z])]
        else:
            for mi in m:
                bcs += [Eq(mi.indexed[t, x, y, 0], 0.)]
                bcs += [Eq(mi.indexed[t, x, y, nz - 1], 0.)]
                bcs += [Eq(mi.indexed[t, x, 0, z], 0.)]
                bcs += [Eq(mi.indexed[t, x, ny - 1, z], 0.)]
                bcs += [Eq(mi.indexed[t, 0, y, z], 0.)]
                bcs += [Eq(mi.indexed[t, nx - 1, y, z], 0.)]

        dx, dy, dz = self.grid_params.d
        dt = self.time_params.d
        subs = {x.spacing: dx, y.spacing: dy, z.spacing: dz, t.spacing: dt}
        op = Operator(bcs + update, subs=subs)

        # Call op trigger compilation
        op(time=1,autotune=True)

        def step(f, t):
            for i, mi in enumerate(m):
                mi.data[(0, ) + self.buffer_slice] = f[i]
            op(time=self.save_every + 1,autotune=True)
            for i, mi in enumerate(m):
                t[i] = mi.data[(self.save_every % 2, ) + self.buffer_slice]

        return step

"""
    def energy_expr(self, m):
        dV = self.grid_params.prod_d
        e = Matrix(self.sim_params.e)
        H = Matrix(self.sim_params.H)
        Kc = dV * -self.sim_params.K
        Ac = dV * self.sim_params.A
        Dc = dV * -self.sim_params.D
        Hc = dV * -self.mu0 * self.sim_params.Ms

        return {"Zeeman":Hc * m.dot(H),
            "Exchange":Ac * vector_gradient(m),
            "Anisotropy":Kc * (m.dot(e))**2,
            "DMI":Dc * m.dot(curl(m))}
    def generate_energy_kernel(self):
        settings = {"shape":self.buffer_dims, "space_order":2}
        m = self.data_matrix(settings)
        energy_expr = self.energy_expr(m)
        E = TimeData(name='E', **settings)
        eqn = Eq(E, sum(energy_expr.values()))
        dx, dy, dz = self.grid_params.d
        subs = {x.spacing: dx, y.spacing: dy, z.spacing: dz}
        # turn dle off because some eqns are 1st and some are 2nd order, requiring different bounds.
        op = Operator(eqn, subs=subs, dle=False)

        # Call op trigger compilation
        op()

        def energy(d):
            for i, mi in enumerate(m):
                mi.data[0] = d[i]
            op(time=1)
            return E.data[0]

        return energy

    def generate_detailed_energy_kernel(self, terms):
        def energy(d):
            settings = {"shape":self.buffer_dims, "space_order":2, "time_dim":len(d), "save":True}
            m = self.data_matrix(settings)
            energy_expr = self.energy_expr(m)

            names = [k for k in terms if k in energy_expr]
            symbols = []
            eqns = []
            for key in names:
                symbol = TimeData(name='E_{}'.format(key), **settings)
                symbols.append(symbol)
                eqns.append(Eq(symbol, energy_expr[key]))

            dx, dy, dz = self.grid_params.d
            subs = {x.spacing: dx, y.spacing: dy, z.spacing: dz}
            # turn dle off because some eqns are 1st and some are 2nd order, requiring different bounds.
            op = Operator(eqns, subs=subs, dle=False)

            for i, mi in enumerate(m):
                for j, dj in enumerate(d):
                    mi.data[j] = dj[i]
            op()
            ret = {}
            for i, name in enumerate(names):
                ret[name] = []
                for dj in symbols[i].data:
                    ret[name].append(dj)
            return ret

        return energy
"""
