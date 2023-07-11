""" class for the traditional geostatistical method ordinary kriging"""
import math
from typing import Union, Tuple, List, Dict

import numpy as np
from matplotlib import pyplot as plt
from numba import jit
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from src.functions import exponential_covariance_function


class OrdinaryKriging:
    """Traditional Geostatistical Method: Ordinary Kriging"""

    def __init__(
        self,
        input_data: np.array,
        output_locations: np.array,
        variogram: Dict,
        anisotropy: Dict = None,
        distance_metric: str = "euclidean",
    ):
        """
        init function for the ordinary kriging class
        :param input_data: np.array with x, y, val
        :param output_locations: np.array with x, y
        :param variogram: must be dictionary with the form of {"nugget": 0.1, "sill": 0.9, "range": 30}
        :param anisotropy: must be dictionary with the form of {"angle": 25, "stretch": 0.1}
        :param distance_metric: str with either euclidean or haversine
        """
        self.c0, self.c1, self.a = (
            variogram["nugget"],
            variogram["sill"],
            variogram["range"],
        )

        self.input_data = input_data
        self.output_locations = output_locations
        self.correct_weights = None
        self.variance = None
        self.estimation = None
        self.output_locations_adjusted = None

        if anisotropy:
            self.angle, self.stretch = (
                anisotropy["angle"] * np.pi / 180,
                anisotropy["stretch"],
            )
            self.apply_anisotropy()
        else:
            self.output_locations_adjusted = output_locations

        self.distance_matrix_input = cdist(
            input_data[:, :2],
            input_data[:, :2],
            metric=self.select_distance_function(distance_metric),
        )

        try:
            self.distance_matrix_output = cdist(
                input_data[:, :2],
                self.output_locations_adjusted,
                metric=self.select_distance_function(distance_metric),
            )
        except ValueError:
            self.distance_matrix_output = cdist(
                input_data[:, :2],
                np.reshape(self.output_locations_adjusted, (-1, 2)),
                metric=self.select_distance_function(distance_metric),
            )

    def apply_anisotropy(self):
        """
        Function to implement to anisotropy concept to the coordinates according to the GSLIB conventions
        https://geostatisticslessons.com/lessons/anglespecification
        """
        stretch_mat = np.array([[1 / self.a, 0], [0, 1 / (self.a * self.stretch)]])

        rotation_mat = np.array(
            [
                [np.cos(self.angle), -np.sin(self.angle)],
                [np.sin(self.angle), np.cos(self.angle)],
            ]
        )
        self.output_locations_adjusted = np.dot(
            stretch_mat, np.dot(rotation_mat, self.output_locations.T)
        ).T
        self.input_data[:, :2] = np.dot(
            stretch_mat, np.dot(rotation_mat, self.input_data[:, :2].T)
        ).T

    def solve_vectorized(self):
        """Function to solve the OK system vectorized"""
        # prepare covariance matrix
        n = self.input_data.shape[0]
        c = np.zeros((n + 1, n + 1))
        c[:n, :n] = exponential_covariance_function(
            self.distance_matrix_input, self.c0, self.c1, self.a
        )
        c[n, :] = 1.0
        c[:, n] = 1.0
        c[n, n] = 0.0

        # inverse covariance matrix
        c_inv = np.linalg.pinv(c)

        # prepare d-matrix
        m = self.output_locations.shape[0]
        d = np.zeros((n + 1, m))
        d[:n, :] = exponential_covariance_function(
            self.distance_matrix_output, self.c0, self.c1, self.a
        )
        d[n, :] = 1

        # calculate weights
        w = np.matmul(c_inv, d)

        # correcting for negative weights
        # if self.correct_weights:  # TODO: include test-function
        #    # see Deutsch, Clayton V. "Correcting for negative weights in ordinary kriging."
        #    w[:n, :][w[:n, :] < 0.01] = 0
        #    d_ = np.mean(d, axis=0)
        #    w_ = np.mean(w, axis=0)
        #    w = np.where(((w > 0) & (d < d_) & (w < w_)), 0, w)
        #    w = np.apply_along_axis(self.scale_sum, 0, w)

        # calculate estimation and variance
        z = self.input_data[:, 2]
        est = np.sum(w[:n, :] * z[:, np.newaxis], axis=0)
        var = self.c0 + self.c1 - (np.sum(w[:n, :] * d[:n, :], axis=0)) + w[n, :]

        return est, var

    def solve_loop(self, num=4):
        """
        Function to solve the OK system with a loop, has the advantage that for the target point the n-nearest points
        can be selected
        :param num: selection of n-nearest points
        """
        # prepare covariance matrix
        n = self.input_data.shape[0]
        c = np.zeros((n + 1, n + 1))
        c[:n, :n] = exponential_covariance_function(
            self.distance_matrix_input, self.c0, self.c1, self.a
        )
        c[n, :] = 1.0
        c[:, n] = 1.0
        c[n, n] = 0.0

        # prepare d-matrix
        m = self.output_locations.shape[0]
        d = np.zeros((n + 1, m))
        d[:n, :] = exponential_covariance_function(
            self.distance_matrix_output, self.c0, self.c1, self.a
        )
        d[n, :] = 1

        est = []
        var = []

        if len(self.output_locations_adjusted.shape) == 1:
            num_of_points = 1
        else:
            num_of_points = self.output_locations_adjusted.shape[0]

        for point_idx in range(num_of_points):
            idx = np.argsort(self.distance_matrix_output[:, point_idx])[
                :num
            ]  # select n nearest points
            idx = np.append(idx, -1)
            c_ = c[idx[:, None], idx]

            # inverse covariance matrix
            c_inv = np.linalg.inv(c_)

            # prepare d-matrix
            d_ = d[:, point_idx][np.ix_(idx)]

            # calculate weights
            w = np.matmul(c_inv, d_)

            # calculate estimation and variance
            z = self.input_data[idx[:num], 2]
            est.append(np.sum(w[:num] * z, axis=0))
            var.append(self.c0 + self.c1 - (np.sum(w[:num] * d_[:num])) + w[num])

        return np.array(est), np.array(var)

    def plot_field(self):
        """plot field fast"""
        plt.tripcolor(
            self.output_locations[:, 0], self.output_locations[:, 1], self.estimation
        )

    def execute(self, backend="vectorized", num=4, exact_values=False):
        """
        Execution of OK interpolation
        :param backend: choose either "vectorized" or "loop"
        :param num: just available within backend loop to select the n-nearest points for specifying the kriging
        equations
        :param exact_values: option to have the exact values for the underlying grid-points
        :return: kriging estimation and kriging variance
        """
        # self.correct_weights = correct_weights
        if backend == "vectorized":
            self.estimation, self.variance = self.solve_vectorized()
        elif backend == "loop":
            self.estimation, self.variance = self.solve_loop(num)

        if exact_values:
            output_locations_tree = cKDTree(self.output_locations_adjusted[:, :2])
            stations_idx = [
                output_locations_tree.query(point[:2])[1] for point in self.input_data
            ]
            self.estimation[stations_idx] = self.input_data[:, 2]

        return self.estimation, self.variance

    @staticmethod
    def start_kriging_for_parallelization(in_val):
        """Function to start kriging parallel"""
        val, out_val, iteration_setting = in_val[0]
        interp = OrdinaryKriging(
            val,
            out_val,
            variogram={
                "nugget": iteration_setting[0],
                "sill": iteration_setting[1],
                "range": iteration_setting[2],
            },
            anisotropy={"angle": iteration_setting[3], "stretch": iteration_setting[4]},
        )
        estimate, _ = interp.execute(backend="vectorized")
        return estimate

    def select_distance_function(self, distance_metric):
        """Function to select or add distance metric"""
        distance_functions = {"haversine": self.haversine, "euclidean": "euclidean"}
        return distance_functions[distance_metric]

    @staticmethod
    @jit(nopython=True)
    def haversine(point_a: Union[Tuple, List], point_b: Union[Tuple, List]) -> float:
        """
        Calculate the great circle distance between two points on the earth (specified in decimal degrees)
        :param point_a: tuple or list with lon/lat value
        :param point_b: tuple or list with lon/lat value
        :return: distance in kilometers
        """
        lon1 = point_a[0]
        lat1 = point_a[1]

        lon2 = point_b[0]
        lat2 = point_b[1]

        # convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

        # haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.asin(math.sqrt(a))
        r = 6372.8  # Radius of earth in kilometers

        return c * r
