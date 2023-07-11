""" Geostatistiscal Simulation with Spectral Turning Bands Method"""
import math
from typing import Dict, List

import numpy as np
from scipy import integrate, interpolate, optimize
from scipy.spatial import cKDTree

from src.ordinary_kriging import OrdinaryKriging


class TurningBands:
    """Traditional Geostatistical Simulation Method: Spectral Turning Bands"""

    def __init__(
            self,
            output_locations: np.array,
            variogram: Dict,
            n_realization: int,
            input_data: np.array = None,
            number_of_lines: int = 1000,
            anisotropy: Dict = None,
    ):
        """
        init function for STBM class
        :param output_locations: np.array with x, y, (z)
        :param variogram: must be dictionary with the form of {"nugget": 0.1, "sill": 0.9, "range": 30}
        :param n_realization: int, number of realization that you wish to create
        :param input_data: input_data: np.array with x, y, val
        :param number_of_lines: int with number used turning bands (lines)
        :param anisotropy: must be dictionary with the form of {"angle": 25, "stretch": 0.1}
        """
        self.output_unconditional_simulation = None
        self.areal = None
        self.real_data = None
        self.c0, self.c1, self.a = (
            variogram["nugget"],
            variogram["sill"],
            variogram["range"],
        )
        self.optimize = optimize
        self.anisotropy = anisotropy
        self.variogram = variogram
        self.input_data = input_data
        self.output_locations = output_locations
        self.n_realization = n_realization
        self.number_of_lines = number_of_lines
        self.interpolation_normal_scores = None
        self.correct_weights = None
        self.variance = None
        self.estimation = None
        self.normal_scores = None
        self.simulation_final = None
        self.transformer = None

        if anisotropy:
            self.angle, self.stretch = anisotropy["angle"], anisotropy["stretch"]
        else:
            self.angle, self.stretch = 0, 1

    def unconditional_simulation(self):
        """Draws an unconditional simulation"""
        np.random.seed()

        r = self.build_rotation_matrix(
            [self.a, self.a * self.stretch, 1, self.angle, 0, 0]
        )

        sims = []

        si = np.arange(0.001, 50, 0.001)

        # time-consuming step, try-except for loading from disk
        try:
            f1_i = np.load("F1i.npy")

        except FileNotFoundError:
            print("not loaded")
            f1_i = np.zeros(si.shape)
            for j, s in enumerate(si):
                f1_i[j] = integrate.quad(
                    self.exponential_f_3, np.finfo(np.float32).eps, s
                )[0]
            f1_i = f1_i / f1_i[-1]
            np.save("F1i.npy", f1_i)

        f = interpolate.interp1d(f1_i, si)

        if self.output_locations.shape[1] == 2:
            self.output_locations = np.c_[
                self.output_locations, np.ones(self.output_locations.shape[0])
            ]

        for _ in range(self.n_realization):
            u_n = f(np.random.rand(self.number_of_lines))
            z_n = self.equal_distributed_lines_over_3d_sphere()
            phi_n = np.random.uniform(0, 2 * np.pi, self.number_of_lines)
            z_1 = z_n @ r.transpose() * u_n.reshape(-1, 1)

            y_sim = np.zeros((self.output_locations.shape[0], self.number_of_lines))

            for i in range(self.number_of_lines):
                y_sim[:, i] = np.sqrt(2 * self.c1) * np.cos(
                    self.output_locations @ z_1[i, :] + phi_n[i]
                )
            y_sim_f = np.sqrt(1 / self.number_of_lines) * y_sim.sum(axis=1)

            final_sim = y_sim_f + np.random.randn(y_sim.shape[0]) * np.sqrt(self.c0)

            sims.append(final_sim)

        self.output_unconditional_simulation = np.stack(sims)

    def conditional_simulation(self):
        """Draws a conditional classical geostatistical simulation, input_data parameter in init class is required"""
        ok = OrdinaryKriging(
            self.input_data.copy(),
            self.output_locations[:, :2].copy(),
            self.variogram,
            self.anisotropy,
        )
        self.interpolation_normal_scores, _ = ok.execute(exact_values=True, backend="vectorized", num=8)

        # unconditional simulation
        self.unconditional_simulation()

        # interpolation of unconditional simulation with stations input
        output_locations_tree = cKDTree(self.output_locations[:, :2])
        stations_idx = [
            output_locations_tree.query(point[:2])[1] for point in self.input_data
        ]
        simulations_stations = self.output_unconditional_simulation[:, stations_idx]
        interpolation_simulation = np.zeros(
            (self.n_realization, self.output_locations.shape[0])
        )

        for s in range(self.n_realization):
            _in = np.c_[self.input_data[:, :2], simulations_stations[s, :]]
            ok = OrdinaryKriging(
                _in,
                self.output_locations[:, :2].copy(),
                self.variogram,
                self.anisotropy,
            )
            interpolation_simulation[s, :], _ = ok.execute(exact_values=True, backend="vectorized", num=8)

        self.simulation_final = self.interpolation_normal_scores + (
                self.output_unconditional_simulation - interpolation_simulation
        )

        return self.simulation_final

    @staticmethod
    def exponential_f_3(s):
        """exponential spectral density function"""
        return (4 * math.pi ** 2 * s ** 2) * (1.0 / (math.pi ** 2 * (1 + s ** 2) ** 2))

    @staticmethod
    def build_rotation_matrix(model: List):
        """
        Set up the matrix to transform the Cartesian coordinates to coordinates that account for ranges and anisotropy.
        The rotation is performed according to the GSLIB conventions.
        https://geostatisticslessons.com/lessons/anglespecification
        :param model:  input must be in following way: [a, a * stretch, 1, angle, 0, 0]
        """
        ranges = model[0:3]
        red_mat = np.diagflat(1 / np.array(ranges))

        alpha, beta, theta = np.deg2rad(
            model[3:6]

        )  # TODO: implement the also the 3-d case!
        # rot_mat = np.zeros((3, 3))
        # rot_mat[0, 0] = cos(alpha) * cos(theta) + sin(alpha) * sin(beta) * sin(theta)
        # rot_mat[0, 1] = -sin(alpha) * cos(theta) + cos(alpha) * sin(beta) * sin(theta)
        # rot_mat[0, 2] = -cos(beta) * sin(theta)
        # rot_mat[1, 0] = sin(alpha) * cos(beta)
        # rot_mat[1, 1] = cos(alpha) * cos(beta)
        # rot_mat[1, 2] = sin(beta)
        # rot_mat[2, 0] = cos(alpha) * sin(theta) - sin(alpha) * sin(beta) * cos(theta)
        # rot_mat[2, 1] = -sin(alpha) * sin(theta) - cos(alpha) * sin(beta) * cos(theta)
        # rot_mat[2, 2] = cos(beta) * cos(theta)

        rot_mat = np.array(
            [
                [np.cos(alpha), -np.sin(alpha), 0],
                [np.sin(alpha), np.cos(alpha), 0],
                [0, 0, 1],
            ]
        )

        return np.linalg.solve(rot_mat, red_mat)

    def equal_distributed_lines_over_3d_sphere(self):
        """
        Generation of equal-distributed lines over the unit 3D sphere according to a van der Corput sequence,
        Returns:
           lines: n*3 matrix representing the lines
        """
        i = np.array([j for j in range(self.number_of_lines)])

        # binary decomposition of i
        j = i
        u = np.zeros(len(i))
        p = 0

        while max(j) > 0:
            p = p + 1
            t = np.floor(j / 2)
            u = u + (2 * (j / 2 - t) / (2 ** p))
            j = t

        # ternary decomposition of i
        j = i
        v = np.zeros(len(i))
        p = 0

        while max(j) > 0:
            p = p + 1
            t = np.floor(j / 3)
            v = v + 3 * (j / 3 - t) / (3 ** p)
            j = t

        # directing vector of the i-th line
        x = np.array(
            [
                np.cos(2 * math.pi * u) * np.sqrt(1 - v ** 2),
                np.sin(2 * math.pi * u) * np.sqrt(1 - v ** 2),
                v,
            ]
        ).T

        # random rotation
        angles = np.random.uniform(0, 360, 3)
        r = self.build_rotation_matrix([1, 1, 1, angles[0], angles[1], angles[2]])
        lines = x @ r

        return lines
