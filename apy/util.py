import json

def write_settings(settings, filename):
    json.dump(settings, open(filename, "w"))

def recurse_to_tuple(data):
    if isinstance(data, list):
        return tuple(data)
    elif isinstance(data, dict):
        return {k:recurse_to_tuple(v) for k,v in data.items()}
    else:
        return data

def read_settings(filename):
    return recurse_to_tuple(json.load(open(filename, "r")))

class ComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj,'reprJSON'):
            return obj.reprJSON()
        elif hasattr(obj, 'tolist'):
            return obj.tolist()
        else:
            return json.JSONEncoder.default(self, obj)

class Struct(object):
    def __init__(self, kwargs):
        for k, v in kwargs.items():
            self.__dict__[k] = Struct(v) if isinstance(v, dict) else v

    def __str__(self):
        ret = ""
        for k, v in self.__dict__.items():
            ret += str(k) + ":" + str(v) + "\n"
        return ret

    def reprJSON(self):
        return self.__dict__
