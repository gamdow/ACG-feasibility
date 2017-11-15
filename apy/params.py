import numpy as np

constants = {
    'mu0':1.2566370614e-6, # permeability of free space
    'gamma0':2.211e5, # gyromagnetic ratio
    }

def dimensions(d=None, l=None, n=None):
    valid_args = {"d":float, "l":float, "n":int}
    args = {k:v for k, v in locals().items() if k in valid_args}
    assert np.sum([1 for v in args.values() if v is not None]) == 2
    max_length = np.max([(len(v) if hasattr(v, "__len__") else 1) for v in args.values() if v is not None])
    args = {k:None if v is None else (np.full(max_length, v, valid_args[k])) for k, v in args.items()}

    if args['d'] is None:
        args['d'] = args['l'] / args['n']
    elif args['l'] is None:
        args['l'] = args['d'] * args['n']
    elif args['n'] is None:
        args['n'] = np.array(args['l'] / args['d'] + 0.5, dtype=int)

    return {k:(tuple(v.tolist()) if hasattr(v, "__len__") and len(v) > 1 else valid_args[k](v)) for k, v in args.items()}

def simulation(grid, time, Ms, alpha, A=0, K=0, D=0, Hmag=0, Hdir=(0,0,1), e=(1,0,0), correction=0, periodic_boundary=True, frames=1, init="random", value=None):
    ret = locals()
    ret['H'] = tuple(Hmag * H for H in Hdir)
    ret.update(constants)  # gyromagnetic ratio
    ret['save_every'] = time['n'] // frames
    return ret

def param(name, type, isterm, min, max, predicate):
    return locals()

param_info = {"A":param("Exchange", "scalar", True, 1e-12, 1e-10, lambda s: s['A'] != 0),
    "K":param("Anisotropy", "scalar", True, 1e2, 1e8, lambda s: s['K'] != 0 and np.linalg.norm(s['e']) != 0),
    "D":param("DMI", "scalar", True, 1e-4, 1e-2, lambda s: s['D'] != 0),
    "Hmag":param("Zeeman", "scalar", True, 1, 1e5, lambda s: s['Hmag'] != 0 and np.linalg.norm(s['Hdir']) != 0),
    "e":param("Anisotropy Axis", "vector", False, 1, 1, lambda s: True),
    "Hdir":param("Zeeman Axis", "vector", False, 1, 1, lambda s: True),
    "Ms":param("Saturation Magnetisation", "scalar", False, 1e5, 1e6, lambda s: True),
    "alpha":param("Gilbert Damping", "scalar", False, 1e-5, 1, lambda s: True)}

def has_term(settings, name):
    d = settings.__dict__ if hasattr(settings, "__dict__") else settings
    for n in param_info.values():
        if n['name'].lower() == name.lower() and n['predicate'](d):
            return True
    return False
