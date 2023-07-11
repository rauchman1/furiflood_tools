import matplotlib.pyplot as plt
import numpy as np

from src.simulation import TurningBands

grid_x = np.arange(1, 100, 1)
grid_y = np.arange(1, 100, 1)
x_vals, y_vals = np.meshgrid(grid_x, grid_y)

coordinates_out = np.c_[x_vals.ravel(), y_vals.ravel()]
n = 5

# isotropic case
simulation = TurningBands(output_locations=coordinates_out,
                          variogram={"nugget": 0.0, "sill": 1.0, "range": 150},
                          n_realization=n)
simulation.unconditional_simulation()
output_sim = simulation.output_unconditional_simulation

fig, ax = plt.subplots(1, n, figsize=(10, 2))
for i in range(n):
    field = np.reshape(output_sim[i, :], x_vals.shape)
    ax[i].pcolormesh(x_vals, y_vals, field)
plt.tight_layout()
plt.show()

# anisotropic case
simulation = TurningBands(output_locations=coordinates_out,
                          variogram={"nugget": 0.0, "sill": 1.0, "range": 100},
                          anisotropy={"angle": 45, "stretch": 0.1},
                          n_realization=n)
simulation.unconditional_simulation()
output_sim = simulation.output_unconditional_simulation

fig, ax = plt.subplots(1, n, figsize=(10, 2))
for i in range(n):
    field = np.reshape(output_sim[i, :], x_vals.shape)
    ax[i].pcolormesh(x_vals, y_vals, field)
plt.tight_layout()
plt.show()

print()
