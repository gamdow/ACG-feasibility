import numpy as np

from . import sim
from .numpy_funcs import vector_laplacian, curl

class Sim(sim.Sim):
    
    framework_name = "NumPy"

    def generate_step_kernel(self):
        assert self.boundary == 1

        m = np.zeros((2, self.ndims) + self.buffer_dims)
        c = 2 / (self.mu0 * self.sim_params.Ms)
        e = np.array(self.sim_params.e)
        H = np.array(self.sim_params.H)[:, np.newaxis, np.newaxis, np.newaxis]
        Kc = c * self.sim_params.K
        Ac = c * self.sim_params.A
        Dc = c * self.sim_params.D
        dx2 = 1 / self.grid_params.d**2
        dx = 1 / (2 * self.grid_params.d)

        anisotropy = (lambda m: Kc * np.multiply.outer(e, np.tensordot(m, e, (0,0)))) if self.sim_params.has_term("anisotropy") else lambda m: 0
        zeeman = (lambda m: H) if self.sim_params.has_term("zeeman") else lambda m: 0
        exchange = (lambda m: Ac * vector_laplacian(m, dx2)) if self.sim_params.has_term("exchange") else lambda m: 0
        dmi = (lambda m: Dc * curl(m, dx)) if self.sim_params.has_term("dmi") else lambda m: 0

        def llg(m):
            heff = anisotropy(m) + zeeman(m) + exchange(m) + dmi(m)
            crossHeff = np.cross(m, heff, axis=0)
            return -self.gamma0 * (crossHeff + self.sim_params.alpha * np.cross(m, crossHeff, axis=0)) / (1 + self.sim_params.alpha**2)

        buffer_slice = (slice(None), ) + self.buffer_slice
        nx, ny, nz = self.buffer_dims
        if self.periodic_boundary:
            def step(f, t):
                m[(0, ) + buffer_slice] = f
                for i in range(self.save_every):
                    b = m[i % 2]
                    b[:,:,:,0] = b[:,:,:,nz-2]
                    b[:,:,0,:] = b[:,:,ny-2,:]
                    b[:,0,:,:] = b[:,nx-2,:,:]
                    b[:,:,:,nz-1] = b[:,:,:,1]
                    b[:,:,ny-1,:] = b[:,:,1,:]
                    b[:,nx-1,:,:] = b[:,1,:,:]
                    m[(i + 1) % 2] = b + self.time_params.d * llg(b)
                t[:] = m[(self.save_every % 2, ) + buffer_slice]
        else:
            def step(f, t):
                m[(0, ) + buffer_slice] = f
                for i in range(self.save_every):
                    b = m[i % 2]
                    b[:,:,:,0] = 0
                    b[:,:,0,:] = 0
                    b[:,0,:,:] = 0
                    b[:,:,:,nz-1] = 0
                    b[:,:,ny-1,:] = 0
                    b[:,nx-1,:,:] = 0
                    m[(i + 1) % 2] = b + self.time_params.d * llg(b)
                t[:] = m[(self.save_every % 2, ) + buffer_slice]

        return step



"""
class MMagNumpy(mmag.MMag):

    def step(self, m):
        if self.correction == "normalise":
            dmdt = self.llg(m)
            new_m = self.dt * dmdt + m
            return normalise(new_m)
        elif self.correction == "v1":
            normm = numpy.sqrt(numpy.sum(m * m, axis=3))
            dmdt = self.llg(m) + numpy.expand_dims(1 - 1 / normm, 3) * m
            return self.dt * dmdt + m
        elif self.correction == "v2":
            dmdt = self.llg(m)
            dmdtc = numpy.abs(dmdt) * numpy.expand_dims(1 - numpy.sum(m * m, axis=3), 3) * m
            return self.dt * (dmdt + self.u * dmdtc) + m
        elif self.correction == "v3":
            dmdt = self.llg(m)
            new_m = self.dt * dmdt + m
            normm = numpy.expand_dims(numpy.sum(new_m * new_m, axis=3), 3)
            dmdtc = -self.alpha**2 * numpy.expand_dims(numpy.sum(new_m * dmdt, axis=3), 3) * new_m / (1 + self.alpha**2 * normm)
            return new_m + self.u * dmdtc * self.dt
        elif self.correction == "v4":
            return self.dt * self.llg2(m) + m
        else:
            return self.dt * self.llg(m) + m

    def exchange_energy(self, idx):
        data = self.data[idx]
        nx, ny, nz, dim = numpy.shape(data)
        exc = numpy.zeros((nx, ny, nz))
        d = 1 / (2 * self.dx)
        for i in range(dim):
            mi = data[:,:,:,i]
            exc[:,:,:] += ((-mi[range(-1, nx - 1),:,:] + mi[range(1 - nx, 1),:,:]) * d[0])**2
            exc[:,:,:] += ((-mi[:,range(-1, ny - 1),:] + mi[:,range(1 - ny, 1),:]) * d[1])**2
            exc[:,:,:] += ((-mi[:,:,range(-1, nz - 1)] + mi[:,:,range(1 - nz, 1)]) * d[2])**2
        return self.A * numpy.sum(exc)

    def prop_energy_metric(self, idx):
        data = self.data[idx]
        return -numpy.sum(data * self.heff(data))

if __name__ == "__main__":
    mgrid = mmag.Grid([1e-6,1,1],[10,1,1],1e-9,100)
    m = MMagNumpy(1e-11,1e5,[0,0,0.2e5],1e5,[1,0,0],1,1,mgrid,1)
    m.step(m.data[0])
"""
