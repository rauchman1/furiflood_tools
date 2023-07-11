import numpy as np
from matplotlib import pyplot as plt

from src.normal_score_transform import normal_score_transform
from src.simulation import TurningBands

# prepare coordinates
grid_x = np.arange(1, 100, 1)
grid_y = np.arange(1, 100, 1)
x_vals, y_vals = np.meshgrid(grid_x, grid_y)
coordinates_out = np.c_[x_vals.ravel(), y_vals.ravel()]
# selection if n realizations
n = 5

# create random "rainfall-like" observational data for conditioning mode
coord_x = np.random.randint(0, 99, 25)
coord_y = np.random.randint(0, 99, 25)
vals = np.random.randint(0, 50, 25)

# create bridge to the gaussian space
transformer, back_transformer = normal_score_transform(vals)
input_data = np.c_[coord_x, coord_y, transformer(vals)]

# initialize simulation class - isotropic case
simulation = TurningBands(
    coordinates_out,
    variogram={"nugget": 0.1, "sill": 0.9, "range": 10},
    n_realization=n,
    input_data=input_data,
    number_of_lines=1000,
)

# conditional simulation with direct back-transformation to original space
output_sim = back_transformer(simulation.conditional_simulation())

# plotting
fig, ax = plt.subplots(1, n, figsize=(10, 2))
for i in range(n):
    field = np.reshape(output_sim[i, :], x_vals.shape)
    m = ax[i].pcolormesh(x_vals, y_vals, field, cmap="Blues")
plt.tight_layout()
plt.colorbar(m)
plt.show()

# initialize simulation class - anisotropic case
simulation = TurningBands(output_locations=coordinates_out,
                          variogram={"nugget": 0.1, "sill": 0.9, "range": 30},
                          anisotropy={"angle": 45, "stretch": 0.3},
                          input_data=input_data,
                          n_realization=n)

# conditional simulation with direct back-transformation to original space
output_sim = back_transformer(simulation.conditional_simulation())

# plotting
fig, ax = plt.subplots(1, n, figsize=(10, 2))
for i in range(n):
    field = np.reshape(output_sim[i, :], x_vals.shape)
    m = ax[i].pcolormesh(x_vals, y_vals, field, cmap="Blues")
plt.tight_layout()
plt.colorbar(m)
plt.show()
