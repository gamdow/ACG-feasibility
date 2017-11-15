import numpy as np
import json
import time
from .util import Struct, ComplexEncoder, recurse_to_tuple

dim_index = 0
n_dims = 3

def buffer_params(settings, boundary):
    if not isinstance(settings, Struct):
        settings = Struct(settings)
    return {'data_dims':settings.grid.n,
        'buffer_dims':tuple(int(n + 2 * boundary) for n in settings.grid.n),
        'buffer_slice':(slice(boundary,-boundary), ) * len(settings.grid.n)}

def norm(val):
    return np.linalg.norm(val, axis=dim_index)

def normalise(val):
    return val / np.expand_dims(norm(val), dim_index)

def set_random(data, value):
    data[:] = np.random.rand(*np.shape(data)) * 2 - 1

def set_vector(data, value):
    data[0] = value[0]
    data[1] = value[1]
    data[2] = value[2]

def set_copy(data, value):
    data[:] = value

def set_disk(data, value=0.7):
    center = np.array(np.shape(data)[1:]) / 2
    rx, ry = center[:2]
    r = min(rx, ry) * value
    data[0:2] = 0
    for index in np.ndindex(data[2].shape):
        d = r - np.linalg.norm(center[:2] - index[:2])
        data[(2,) + index] = 1 if d >= 0 else -1

def set_vortex(data, value):
    center = np.array(np.shape(data)[1:]) / 2
    for index in np.ndindex(data[2].shape):
        x, y, z = center - index
        data[(slice(None),) + index] = y, -x, z

def set_flower(data, value):
    center = np.array(np.shape(data)[1:]) / 2
    for index in np.ndindex(data[2].shape):
        if (center == index).all():
            data[(slice(None),) + index] = [0,0,1]
        else:
            data[(slice(None),) + index] = (center - index) * -1

init_functions = {"random":set_random,
    "vector":set_vector,
    "copy":set_copy,
    "disk":set_disk,
    "vortex":set_vortex,
    "flower":set_flower}

class Data(object):
    def __init__(self, settings, framework, threads):
        settings = settings.__dict__ if hasattr(settings, "__dict__") else settings
        self.settings = settings
        self.framework = framework
        self.threads = threads
        self.data_dims = (n_dims,) + buffer_params(settings, 0)['data_dims']
        self.times = [0]
        self.data = [np.zeros(self.data_dims)]
        self.start_time = None
        self.run_time = None
        init_functions[settings['init']](self.data[0], settings['value'])
        self.data[0] = normalise(self.data[0])

    def reprJSON(self):
        return {"settings":self.settings,
            "framework":self.framework,
            "threads":self.threads,
            "run_time":self.run_time,
            "data_dims":self.data_dims,
            "times":self.times}

    def dump(self, settings_file, data_file=None):
        json.dump(self, open(settings_file, "w"), cls=ComplexEncoder)
        if not data_file is None:
            json.dump(self.data, open(data_file, "w"), cls=ComplexEncoder)

    def load(self, settings_file, data_file=None):
        try:
            data = json.load(open(settings_file, "r"))
        except IOError:
            print("*** Failure ***: File {} does not exist".format(settings_file))
            return False
        if data['framework'].lower() != self.framework.lower():
            print("*** Failure ***: Mismatched framework expected {} got {}".format(self.framework, data['framework']))
            return False
        if data['threads'] != self.threads:
            print("*** Failure ***: Mismatched threads expected {} got {}".format(self.threads, data['threads']))
            return False
        settings = recurse_to_tuple(data['settings'])
        if settings != self.settings:
            print("*** Failure ***: Mismatched settings {} vs {}".format(settings, self.settings))
            return False
        self.run_time = data['run_time']
        self.data_dims = data['data_dims']
        self.times = data['times']
        if not data_file is None:
            try:
                data = json.load(open(data_file, "r"))
            except IOError:
                print("*** Failure ***: File {} does not exist".format(data_file))
                return False
            self.data = [np.array(d) for d in data]
        return True

    def push(self, time):
        self.times.append(time)
        self.data.append(np.zeros(self.data_dims))

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def start_timer(self):
        self.start_time = time.time()

    def end_timer(self):
        self.run_time = time.time() - self.start_time
