import numpy as np
import os
import sys
import apy
import subprocess
import progressbar
import json
import apy.plotting as plt

configs = {
    'steps':[1,100,1000]
    , 'grid':[(100,100,100),(316,316,10),(316,10,316),(10,316,316)]
    , 'threads':[2**i for i in range(3, -1, -1)]
    , 'sim_params':[{'A':1e-11, 'K':0.1e3, 'D':0, 'Hmag':0.1e3}]
}
frames = 1

test_list = []
for idx in np.ndindex(*[len(v) for v in configs.values()]):
    test_list.append({k:v[i] for (k, v), i in zip(configs.items(), idx)})

classes = ['oommf', 'devito', 'opensbli']

num_runs_per_test = 1

bar = progressbar.ProgressBar(max_value=num_runs_per_test * len(test_list) * len(classes))
bar.update(0)

try:
    results = json.load(open("temp/results.json", "r"))
except IOError:
    results = []

for test_idx, test in enumerate(test_list):

    settings = apy.sim_params(
        grid=apy.dim_params(n=test['grid'], d=1e-9),
        time=apy.dim_params(d=1e-14, n=test['steps']),
        Ms=8e5, alpha=0.8,
        **test['sim_params'],
        frames=1,
        init="flower")
    apy.write_settings(settings, "temp/settings.json")
    print("\n")
    print(settings)

    with open("temp/env.list", "w") as envfile:
        print('OMP_NUM_THREADS={}'.format(test['threads']), file=envfile)
        print('OOMMF_THREADS={}'.format(test['threads']), file=envfile)
    print(open("temp/env.list", "r").read())

    for run_idx in range(num_runs_per_test):
        for class_idx, class_name in enumerate(classes):
            print("\n" + class_name + "...")
            proc = subprocess.Popen(["make", "wrap_{}".format(class_name)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=os.environ)
            stdout, stderr = proc.communicate()

            failed = False
            if proc.returncode != 0:
                failed = True
            else:
                data = apy.Data(settings, class_name, test['threads'])
                failed = not data.load("temp/data.json")

            filename = "g{}_s{}_t{}_{}".format(settings['grid']['n'], settings['frames'] * settings['save_every'], test['threads'], class_name).replace(",", "").replace(" ", "_")

            if failed:
                print("\nFailed")
                sys.stdout.flush()
                errfile = "temp/" + filename + ".stderr"
                with open(errfile, "w") as outfile:
                    print(stdout.decode('utf-8'), file=outfile)
                    print(stderr.decode('utf-8'), file=outfile)
                print("output sent to " + errfile)

            else:
                output = data.reprJSON()
                del output['data_dims']
                del output['times']
                print("\nSuccess ({}s)".format(output['run_time']))
                sys.stdout.flush()
                results.append(output)
                outfile = "temp/" + filename
                plt.frame(data[-1], settings['grid']['n'][2] // 2, settings['grid']['l'], data.times[-1], image_width=5, filename=outfile, show_plot=False)
                print("output sent to " + outfile + ".png")
                json.dump(results, open("temp/results.json", "w"))

            bar.update(1 + class_idx + len(classes) * (run_idx + num_runs_per_test * test_idx))
