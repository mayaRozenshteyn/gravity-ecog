import os
import sys
import numpy as np
import iaaft

idx = int(os.environ["SLURM_ARRAY_TASK_ID"])

luminance_g2 = np.load(os.path.join(sys.path[0], "luminance_g2.npy"), allow_pickle = True)

surrog = iaaft.surrogates(luminance_g2, ns=1)

np.save(os.path.join(sys.path[0], "LumSurrog/shuffled-luminance2-" + str(idx) + ".npy"), surrog)