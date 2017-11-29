import os
import sys

if not "DEVITO_OPENMP" in os.environ or os.environ["DEVITO_OPENMP"] != "1":
    print("*** WARNING: Devito OpenMP environment variable has not been set ***", file=sys.stderr)


import numpy as np
from sympy import Matrix, Eq, solve
import progressbar
from devito import Grid, Function, TimeFunction, configuration, Operator

from . import sim, dim_params

configuration['log_level'] = 'WARNING'

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

def VectorTimeFunction(name, settings):
    return Matrix([TimeFunction(name='{}_x'.format(name), **settings), TimeFunction(name='{}_y'.format(name), **settings), TimeFunction(name='{}_z'.format(name), **settings)])

def VectorDenseData(name, settings):
    return Matrix([Function(name='{}_x'.format(name), **settings), Function(name='{}_y'.format(name), **settings), Function(name='{}_z'.format(name), **settings)])

expression_cache = {}
# bit different from normal RK matrix, rather than k1, k2, k3 it's kn-1, kn-2, kn-3
#RKc = [[0, 0, 0], [0.5, 0, 0], [1, 2, 0], [1/6, 2/3, 1/6]]
RKc = [[0, 0, 0], [0.5, 0, 0], [2, 1, 0], [1/6, 2/3, 1/6]]

class Sim(sim.Sim):

    framework_name = "Devito"

    def __init__(self, sim_params, grid_params, time_params, **kwargs):
        scaled_d = 2 * time_params.d
        rktp = dim_params.DimParams(d=scaled_d, l=time_params.l)
        kwargs['save_every'] = int(kwargs['save_every'] * rktp.n / time_params.n)
        super(Sim, self).__init__(sim_params, grid_params, rktp, **kwargs)

    @property
    def data_shape(self):
        # Devito doesn't like numpy types for the grid dimensions, and it needs to be a tuple, so shape needs to be converted
        return tuple(int(i) for i in self.grid_params.n)

    def generate_step_kernel(self):

        print(tuple(self.grid_params.l), file=sys.stderr)

        grid = Grid(shape=self.buffer_dims, extent=tuple(self.grid_params.l))

        settings = {"grid":grid, "space_order":2}
        m = VectorTimeFunction('m', settings)

        c = 2 / (self.mu0 * self.sim_params.Ms)
        zeeman = Matrix(self.sim_params.H)
        exchange = self.sim_params.A * c * vector_laplacian(m)
        e = Matrix(self.sim_params.e)
        anisotropy = self.sim_params.K * c * m.dot(e) * e
        dmi = self.sim_params.D * c * curl(m)
        heff = zeeman + exchange + anisotropy + dmi
        crossHeff = m.cross(heff)
        dmdt_rhs = -self.gamma0 / (1 + self.sim_params.alpha**2) * (crossHeff + self.sim_params.alpha * m.cross(crossHeff))
        dmdt_lhs = VectorTimeFunction('dmdt', settings)
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
        x, y, z = grid.dimensions
        t = grid.stepping_dim
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
        op = Operator(bcs + update, dse=None)
        print(op.ccode, file=sys.stderr)

        # Call op trigger compilation
        op(time=1, dt=self.time_params.d)

        def step(f, t):
            for i, mi in enumerate(m):
                mi.data[(0, ) + self.buffer_slice] = f[i]
            op(time=self.save_every + 1, dt=self.time_params.d, autotune=True)
            for i, mi in enumerate(m):
                t[i] = mi.data[(self.save_every % 2, ) + self.buffer_slice]

        return step

    """
    def generate_step_kernel(self):
        m = VectorTimeData('m', {"shape":self.buffer_dims, "space_order":2, "time_order":3})
        k = VectorTimeData('k', {"shape":self.buffer_dims, "space_order":2, "time_order":3})
        kc = TimeData(name='kc', **{"shape":(3,), "space_order":1, "time_order":3})

        kc.data[:] = np.array(RKc)

        nx, ny, nz = self.buffer_dims
        dt = self.time_params.d
        update = []

        c = 2 / (self.mu0 * self.sim_params.Ms)
        e = Matrix(self.sim_params.e)

        zeeman = Matrix(self.sim_params.H)
        exchange = self.sim_params.A * c * vector_laplacian(m)
        anisotropy = self.sim_params.K * c * m.dot(e) * e
        dmi = self.sim_params.D * c * curl(m)
        heff = zeeman + exchange + anisotropy + dmi

        crossHeff = m.cross(heff)
        LLG = -self.gamma0 / (1 + self.sim_params.alpha**2) * (crossHeff + self.sim_params.alpha * m.cross(crossHeff))

        for ki, llgi in zip(k, LLG):
            update.append(Eq(ki.indexed[t + 1, x, y, z], llgi))
            if self.periodic_boundary:
                update.append(Eq(ki.indexed[t + 1, x, y, 0], ki.indexed[t + 1, x, y, nz - 2]))
                update.append(Eq(ki.indexed[t + 1, x, y, nz - 1], ki.indexed[t + 1, x, y, 1]))
                update.append(Eq(ki.indexed[t + 1, x, 0, z], ki.indexed[t + 1, x, ny - 2, z]))
                update.append(Eq(ki.indexed[t + 1, x, ny - 1, z], ki.indexed[t + 1, x, 1, z]))
                update.append(Eq(ki.indexed[t + 1, 0, y, z], ki.indexed[t + 1, nx - 2, y, z]))
                update.append(Eq(ki.indexed[t + 1, nx - 1, y, z], ki.indexed[t + 1, 1, y, z]))
            else:
                update.append(Eq(ki.indexed[t + 1, x, y, 0], 0.))
                update.append(Eq(ki.indexed[t + 1, x, y, nz - 1], 0.))
                update.append(Eq(ki.indexed[t + 1, x, 0, z], 0.))
                update.append(Eq(ki.indexed[t + 1, x, ny - 1, z], 0.))
                update.append(Eq(ki.indexed[t + 1, 0, y, z], 0.))
                update.append(Eq(ki.indexed[t + 1, nx - 1, y, z], 0.))

        i = 0
        for mi, ki in zip(m, k):
            update.append(Eq(mi.indexed[t + 1, x, y, z], mi.indexed[t, x, y, z] + kc.indexed[t + 1, 0] * ki.indexed[t + 1, x, y, z] + kc.indexed[t, 1] * ki.indexed[t, x, y, z] + kc.indexed[t - 1, 2] * ki.indexed[t - 1, x, y, z]))

        dx, dy, dz = self.grid_params.d
        subs = {x.spacing: dx, y.spacing: dy, z.spacing: dz, t.spacing: dt}
        op = Operator(update, subs=subs)
        print(op.ccode, file=sys.stderr)

        # Call op trigger compilation
        op(time=1)

        def step(f, t):
            for i, mi in enumerate(m):
                mi.data[(0, ) + self.buffer_slice] = f[i]
            op(time=self.save_every + 1)
            for i, mi in enumerate(m):
                t[i] = mi.data[(self.save_every % 2, ) + self.buffer_slice]

        return step

    RK, does not optimise
    def generate_step_kernel(self):
        settings = {"shape":self.buffer_dims, "space_order":2}
        m = VectorTimeData('m', settings)

        k = []
        temp = []
        for i in range(len(RKc) - 1):
            k.append(VectorDenseData('k{}'.format(i + 1), settings))
            temp.append(VectorDenseData('temp{}'.format(i + 1), settings))

        nx, ny, nz = self.buffer_dims
        dt = self.time_params.d
        update = []


        for i, (ki, ti) in enumerate(zip(k, temp)):
            for j, mj in enumerate(m):
                update.append(Eq(ti[j], mj + dt * Matrix(RKc[i]).dot(Matrix([kk[j] for kk in k]))))

            if self.periodic_boundary:
                for mi in ti:
                    update.append(Eq(mi.indexed[x, y, 0], mi.indexed[x, y, nz - 2]))
                    update.append(Eq(mi.indexed[x, y, nz - 1], mi.indexed[x, y, 1]))
                    update.append(Eq(mi.indexed[x, 0, z], mi.indexed[x, ny - 2, z]))
                    update.append(Eq(mi.indexed[x, ny - 1, z], mi.indexed[x, 1, z]))
                    update.append(Eq(mi.indexed[0, y, z], mi.indexed[nx - 2, y, z]))
                    update.append(Eq(mi.indexed[nx - 1, y, z], mi.indexed[1, y, z]))
            else:
                for mi in ti:
                    update.append(Eq(mi.indexed[x, y, 0], 0.))
                    update.append(Eq(mi.indexed[x, y, nz - 1], 0.))
                    update.append(Eq(mi.indexed[x, 0, z], 0.))
                    update.append(Eq(mi.indexed[x, ny - 1, z], 0.))
                    update.append(Eq(mi.indexed[0, y, z], 0.))
                    update.append(Eq(mi.indexed[nx - 1, y, z], 0.))


            c = 2 / (self.mu0 * self.sim_params.Ms)
            e = Matrix(self.sim_params.e)
            zeeman = Matrix(self.sim_params.H)
            exchange = self.sim_params.A * c * vector_laplacian(ti)
            anisotropy = self.sim_params.K * c * ti.dot(e) * e
            dmi = self.sim_params.D * c * curl(ti)
            heff = zeeman + exchange + anisotropy + dmi
            crossHeff = ti.cross(heff)
            LLG = -self.gamma0 / (1 + self.sim_params.alpha**2) * (crossHeff + self.sim_params.alpha * ti.cross(crossHeff))

            for j, kij in enumerate(ki):
                update.append(Eq(kij, LLG[j]))


        for i, mi in enumerate(m):
            update.append(Eq(mi.forward, solve(mi.dt - Matrix(RKc[-1]).dot(Matrix([kj[i] for kj in k])), mi.forward)[0]))

        dx, dy, dz = self.grid_params.d
        subs = {x.spacing: dx, y.spacing: dy, z.spacing: dz, t.spacing: dt}
        op = Operator(update, subs=subs)
        print(op.ccode, file=sys.stderr)

        # Call op trigger compilation
        op(time=1)

        def step(f, t):
            for i, mi in enumerate(m):
                mi.data[(0, ) + self.buffer_slice] = f[i]
            op(time=self.save_every + 1)
            for i, mi in enumerate(m):
                t[i] = mi.data[(self.save_every % 2, ) + self.buffer_slice]

        return step

    def generate_step_kernel(self):
        settings = {"shape":self.buffer_dims, "space_order":2}
        m = VectorData('m', settings)
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
        print(op.ccode, file=sys.stderr)

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
