import os

import sys
import numpy as np
import progressbar
import discretisedfield as df
import shutil
import subprocess
import glob
import sarge

from . import sim

grid_json = \
"""Specify Oxs_BoxAtlas:WorldAtlas {{
    xrange {{0 {} }}
    yrange {{0 {} }}
    zrange {{0 {} }}
}}

"""

mesh_bounded_json = \
"""Specify Oxs_RectangularMesh:mesh {{
  cellsize {{ {} {} {} }}
  atlas :WorldAtlas
}}

"""

mesh_periodic_json = \
"""Specify Oxs_PeriodicRectangularMesh:mesh {{
  cellsize {{ {} {} {} }}
  atlas :WorldAtlas
  periodic "xyz"
}}

"""

solver_json = \
"""Specify Oxs_FileVectorField:m0file {{
   atlas :WorldAtlas
   file m0.omf
}}

Specify Oxs_EulerEvolve {{
  alpha {alpha}
  gamma_G {gamma}
  min_timestep {timestep}
  max_timestep {timestep}
}}

Specify Oxs_TimeDriver {{
  evolver Oxs_EulerEvolve
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

"""

zeeman_json = \
"""Specify Oxs_FixedZeeman {{
  field {{
    Oxs_UniformVectorField {{
      vector {{ {} {} {} }}
    }}
  }}
  multiplier 1
}}

"""

anisotropy_json = \
"""Specify Oxs_UniaxialAnisotropy {{
  K1 {}
  axis {{{} {} {}}}
 }}

"""

exchange_json = \
"""Specify Oxs_UniformExchange {{
  A {}
}}

"""

dmi_json = \
"""Specify Oxs_BulkDMI:DMEx [subst {{
  default_D {D}
  atlas :WorldAtlas
  D {{
    WorldAtlas WorldAtlas {D}
  }}
}}]

"""

class Sim(sim.Sim):

    framework_name = "OOMMF"
    path = "oommf_temp"

    def generate_step_kernel(self):

        if self.periodic_boundary and self.sim_params.has_term("DMI"):
            raise NotImplementedError

        mif = "# MIF 2.1\n\n"

        mif += grid_json.format(*self.grid_params.l)

        mesh_json = mesh_periodic_json if self.periodic_boundary else mesh_bounded_json
        mif += mesh_json.format(*self.grid_params.d)

        terms = {"Zeeman":zeeman_json.format(*self.sim_params.H),
            "Exchange":exchange_json.format(self.sim_params.A / self.sim_params.Ms),
            "Anisotropy":anisotropy_json.format(self.sim_params.K / self.sim_params.Ms, *self.sim_params.e),
            "DMI":dmi_json.format(D=self.sim_params.D / self.sim_params.Ms)}
        for i in [v for k, v in terms.items() if k in self.sim_params.terms]:
            mif += i

        mif += solver_json.format(alpha=self.sim_params.alpha, gamma=self.gamma0, timestep=self.time_params.d, time=self.time_params.d * self.save_every)

        shutil.rmtree(self.path, ignore_errors=True)
        os.mkdir(self.path)
        mif_file = open("{}/run.mif".format(self.path), "w")
        mif_file.write(mif)
        mif_file.close()

        def step(f, t):
            mesh = df.Mesh(p1=(0, 0, 0), p2=self.grid_params.l, cell=self.grid_params.d)
            field = df.Field(mesh, value=(0,0,1), norm=1)
            field.array[:] = np.transpose(f, (1,2,3,0))
            #field.norm = self.sim_params.Ms
            field.write("{}/m0.omf".format(self.path))

            out = subprocess.Popen("tclsh $OOMMFTCL boxsi +fg run.mif", cwd="./{}".format(self.path), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = out.communicate()
            if out.returncode:
                if not stdout is None:
                    print(stdout.decode('utf-8'), file=sys.stderr)
                if not stderr is None:
                    print(stderr.decode('utf-8'), file=sys.stderr)
                sys.stderr.flush()
                raise RuntimeError

            last_omf_file = max(glob.iglob("{}/system*.omf".format(self.path)), key=os.path.getctime)
            field = df.read(last_omf_file)
            #field.norm = 1
            t[:] = np.transpose(field.array, (3,0,1,2))

        return step
