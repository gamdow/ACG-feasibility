import numpy as np

class DimParams(object):
    def __init__(self, d=None, l=None, n=None, bounded=True):
        self.bounded = bounded
        if bounded:
            args = {"d":float, "l":float, "n":int}

            valid_locals = [(k, v) for k, v in locals().items() if (k in args and v != None)]

            known = 0
            max_length = 0
            for key, val in valid_locals:
                known += 1
                max_length = max(max_length, len(val) if hasattr(val, "__len__") else 1)
            assert(known == 2) # need 2 out of 3

            def arrayify(val, typ):
                if max_length == 1:
                    return typ(val)
                elif hasattr(val, "__len__"):
                    return np.array(val, dtype=typ)
                else:
                    return np.full(max_length, val, dtype=typ)

            for key, val in valid_locals:
                self.__dict__[key] = arrayify(val, args[key])

            if d == None:
                self.d = self.l / self.n
            elif l == None:
                self.l = self.d * self.n
            elif n == None:
                self.n = arrayify(self.l / self.d + 0.5, int)
        else:
            assert(d is not None)
            self.d = float(d)

    @property
    def prod_d(self):
        return np.prod(self.d)

    @property
    def prod_n(self):
        return np.prod(self.n)

    @property
    def prod_l(self):
        return np.prod(self.l)

    def __str__(self):
        if self.bounded:
            return "\td = {}\n\tl = {}\n\tn = {}".format(self.d, self.l, self.n)
        else:
            return "\td = {}".format(self.d)

    @property
    def n_as_tuple(self):
        return self.n if hasattr(self.n, "__iter__") else (self.n,)
