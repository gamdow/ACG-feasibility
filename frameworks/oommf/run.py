import os
import sys

if len(sys.argv) < 3:
    print("*** ERROR: Missing input and output filenames ***", file=sys.stderr)
    sys.exit()
input_file = str(sys.argv[1])
output_file = str(sys.argv[2])

import numpy as np
import shutil
import glob
import sarge
import discretisedfield as df
import progressbar
import subprocess
import apy

tcl = {"grid":
"""Specify Oxs_BoxAtlas:WorldAtlas {{
    xrange {{0 {} }}
    yrange {{0 {} }}
    zrange {{0 {} }}
}}

""",

"mesh_bounded":
"""Specify Oxs_RectangularMesh:mesh {{
  cellsize {{ {} {} {} }}
  atlas :WorldAtlas
}}

""",

"mesh_periodic":
"""Specify Oxs_PeriodicRectangularMesh:mesh {{
  cellsize {{ {} {} {} }}
  atlas :WorldAtlas
  periodic "xyz"
}}

""",

"solver":
"""Specify Oxs_FileVectorField:m0file {{
   atlas :WorldAtlas
   file m0.omf
}}

Specify Oxs_RungeKuttaEvolve:evolver {{
  alpha {alpha}
  gamma_G {gamma}
}}

Specify Oxs_TimeDriver {{
  evolver evolver
  stopping_time {time}
  mesh :mesh
  vector_field_output_format {{ text %.17g }}
  scalar_output_format %.17g
  stage_count 1
  Ms {{
    Oxs_VecMagScalarField {{
      field :m0file
    }}
  }}
  m0 :m0file
  basename system
}}

Destination archive mmArchive
Schedule Oxs_TimeDriver::Magnetization archive stage 1

""",

"zeeman":
"""Specify Oxs_FixedZeeman {{
  field {{
    Oxs_UniformVectorField {{
      vector {{ {} {} {} }}
    }}
  }}
  multiplier 1
}}

""",

"anisotropy":
"""Specify Oxs_UniaxialAnisotropy {{
  K1 {}
  axis {{{} {} {}}}
 }}

""",

"exchange":
"""Specify Oxs_UniformExchange {{
  A {}
}}

""",

"dmi":
"""Specify Oxs_BulkDMI:DMEx [subst {{
  default_D {D}
  atlas :WorldAtlas
  D {{
    WorldAtlas WorldAtlas {D}
  }}
}}]

"""
}

settings = apy.Struct(apy.read_settings(input_file))
buffer_params = apy.Struct(apy.buffer_params(settings, 1))

if settings.periodic_boundary and apy.has_term(settings, "DMI"):
    raise NotImplementedError

mif = "# MIF 2.1\n\n"
mif += tcl['grid'].format(*settings.grid.l)
mif += tcl['mesh_periodic' if settings.periodic_boundary else 'mesh_bounded'].format(*settings.grid.d)

terms = {"Zeeman":tcl['zeeman'].format(*settings.Hdir),
    "Exchange":tcl['exchange'].format(settings.A / settings.Ms),
    "Anisotropy":tcl['anisotropy'].format(settings.K / settings.Ms, *settings.e),
    "DMI":tcl['dmi'].format(D=settings.D / settings.Ms)}
for i in [v for k, v in terms.items() if apy.has_term(settings, k)]:
    mif += i

mif += tcl['solver'].format(alpha=settings.alpha, gamma=settings.gamma0, timestep=settings.time.d, time=settings.time.d * settings.save_every)

print("Generating mif ...", file=sys.stderr)
mif_file = open("run.mif", "w")
mif_file.write(mif)
mif_file.close()

def step(f, t):
    mesh = df.Mesh(p1=(0, 0, 0), p2=settings.grid.l, cell=settings.grid.d)
    field = df.Field(mesh, value=(0,0,1), norm=1)
    field.array[:] = np.transpose(f, (1,2,3,0))
    #field.norm = self.sim_params.Ms
    field.write("m0.omf")

    out = subprocess.Popen("tclsh $OOMMFTCL boxsi +fg run.mif", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = out.communicate()
    if out.returncode:
        if not stdout is None:
            print(stdout.decode('utf-8'), file=sys.stderr)
        if not stderr is None:
            print(stderr.decode('utf-8'), file=sys.stderr)
        sys.stderr.flush()
        raise RuntimeError

    last_omf_file = max(glob.iglob("system*.omf"), key=os.path.getctime)
    field = df.read(last_omf_file)
    #field.norm = 1
    t[:] = np.transpose(field.array, (3,0,1,2))

num_threads = int(os.environ['OOMMF_THREADS']) if 'OOMMF_THREADS' in os.environ else 1
data = apy.Data(settings, "OOMMF", num_threads)

print("Running simulation ...", file=sys.stderr)
data.start_timer()
for frame in range(settings.frames):
    data.push((frame + 1) * settings.time.d * settings.save_every)
    step(data[-2], data[-1])
data.end_timer()

data.dump(output_file)
