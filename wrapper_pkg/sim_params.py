import sys
import numpy as np

def param(name, type, isterm, min, max, predicate):
    return locals()

param_info = {"A":param("Exchange", "scalar", True, 1e-12, 1e-10, lambda s: s.A != 0),
    "K":param("Anisotropy", "scalar", True, 1e2, 1e8, lambda s: s.K != 0 and np.linalg.norm(s.e) != 0),
    "D":param("DMI", "scalar", True, 1e-4, 1e-2, lambda s: s.D != 0),
    "Hmag":param("Zeeman", "scalar", True, 1, 1e5, lambda s: s.Hmag != 0 and np.linalg.norm(s.Hdir) != 0),
    "e":param("Anisotropy Axis", "vector", False, 1, 1, lambda s: True),
    "Hdir":param("Zeeman Axis", "vector", False, 1, 1, lambda s: True),
    "Ms":param("Saturation Magnetisation", "scalar", False, 1e5, 1e6, lambda s: True),
    "alpha":param("Gilbert Damping", "scalar", False, 1e-5, 1, lambda s: True)}

heff_names = [n['name'] for n in param_info.values() if n['isterm']]

class SimParams(object):

    class Info(object):
        def __init__(self, **kwargs):
            self.__dict__.update(**kwargs)

    def parse_info(self, key, val):
        info = param_info[key]
        info.update({'value':val, 'norm':np.linalg.norm(val)})
        return self.Info(**info)

    def string_info(self, key, val):
        info = self.parse_info(key, val)
        if info.norm != 0:
            return info.isterm, "\t{} ({}) = {}\n".format(info.name, key, val)
        else:
            return info.isterm, ""

    def param_validate(self, key, val):
        info = self.parse_info(key, val)
        if info.norm != 0 and (info.norm < info.min or info.norm > info.max):
            print((info.name + ", {}, outside expected range [{:.1e},{:.1e}]").format(val, info.min, info.max), file=sys.stderr)

    def has_term(self, name):
        for n in param_info.values():
            if n['name'].lower() == name.lower() and n['predicate'](self):
                return True
        return False

    @property
    def terms(self):
        return [n for n in heff_names if self.has_term(n)]

    def __init__(self, Ms, alpha, A=0, K=0, D=0, Hmag=0, Hdir=[0,0,0], e=[0,0,0]):
        for key, val in [(k, v) for k, v in locals().items() if k in param_info]:
            self.param_validate(key, val)
            self.__dict__[key] = val
        self.H = [i * Hmag for i in Hdir]
        assert sum([self.has_term(name) for name in heff_names]) > 0, "Simulation parameters will produce a effective field without any terms."

    def __str__(self):
        heff = ""
        aux = ""
        for key in [k for k in param_info if k in self.__dict__]:
            isterm, string = self.string_info(key, self.__dict__[key])
            if isterm:
                heff += string
            else:
                aux += string

        return "EFFECTIVE FIELD TERM PARAMETERS\n{}\nAUXIALLARY PARAMETERS\n{}".format(heff, aux)
