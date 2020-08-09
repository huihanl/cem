import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import sys
import os

elite_path = sys.argv[1]

fig = plt.figure()

elites = np.load(elite_path, allow_pickle=True)

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)

    if not os.path.exists(directory):
        os.makedirs(directory)

for i in range(12):
    ax = plt.axes(projection='3d')
    for z in elites:
        this_z = z[i*4:i*4+4]
        ax.scatter3D(this_z[0], this_z[1], this_z[2])

    z_dir = os.path.join("test_z", elite_path[:-4])
    ensure_dir(z_dir)
    fig.savefig(os.path.join(z_dir, "z_action_{}.png".format(i)))
    plt.clf()
