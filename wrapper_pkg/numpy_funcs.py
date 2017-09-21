import numpy as np

bounds = slice(1,-1)
dim = (slice(None),)
mid = (bounds, bounds, bounds)
xl = (slice(0,-2), bounds, bounds)
xr = (slice(2,None), bounds, bounds)
yl = (bounds, slice(0,-2), bounds)
yr = (bounds, slice(2,None), bounds)
zl = (bounds, bounds, slice(0,-2))
zr = (bounds, bounds, slice(2,None))

def vector_laplacian(u, dx2):
    ret = np.zeros_like(u)
    ret[dim + mid] = (u[dim + xr] + u[dim + xl]) * dx2[0] + (u[dim + yr] + u[dim + yl]) * dx2[1] + (u[dim + zl] + u[dim + zr]) * dx2[2] - 2 * u[dim + mid] * (dx2[0] + dx2[1] + dx2[2])
    return ret

def vector_gradient(u, dx):
    ret = np.zeros_like(u[0])
    ret[mid] += np.sum(((u[dim + xr] - u[dim + xl]) * dx[0])**2, axis=0)
    ret[mid] += np.sum(((u[dim + yr] - u[dim + yl]) * dx[1])**2, axis=0)
    ret[mid] += np.sum(((u[dim + zr] - u[dim + zl]) * dx[2])**2, axis=0)
    return ret

def curl(u, dx):
    ret = np.zeros_like(u)
    ret[(0,) + mid] = (u[(2,) + yr] - u[(2,) + yl]) * dx[1] - (u[(1,) + zr] - u[(1,) + zl]) * dx[2]
    ret[(1,) + mid] = (u[(0,) + zr] - u[(0,) + zl]) * dx[2] - (u[(2,) + xr] - u[(2,) + xl]) * dx[0]
    ret[(2,) + mid] = (u[(1,) + xr] - u[(1,) + xl]) * dx[0] - (u[(0,) + yr] - u[(0,) + yl]) * dx[1]
    return ret

def haloswap(data, nx, ny, nz, b):
    ix, iy, iz = nx - 1, ny - 1, nz - 1
    data[:,:,:,0:b] = data[:,:,:,nz-2*b:nz-b]
    data[:,:,0:b,:] = data[:,:,ny-2*b:ny-b,:]
    data[:,0:b,:,:] = data[:,nx-2*b:nx-b,:,:]
    data[:,:,:,nz-b:nz] = data[:,:,:,b:2*b]
    data[:,:,ny-b:ny,:] = data[:,:,b:2*b,:]
    data[:,nx-b:nx,:,:] = data[:,b:2*b,:,:]
