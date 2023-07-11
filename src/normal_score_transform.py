"""Function for normal score transformation"""
from typing import Tuple

import numpy as np
from scipy.interpolate import interp1d, interpolate


def normal_score_transform(
        data_input: np.array,
) -> Tuple[interp1d, interp1d]:
    """
    This function transforms any distribution into the normal space
    :param data_input: values to be transformed
    :return: 1D-interpolator for transformation and back-transformation
    """

    # create transformation with order and input values
    transformation_table = np.c_[np.arange(len(data_input)), data_input]

    # sort observed values
    transformation_table = transformation_table[transformation_table[:, 1].argsort()]

    # add calculated probability for a given normal distribution value = normal scores
    transformation_table = np.c_[
        transformation_table,
        np.sort(
            np.random.normal(0, 1, len(data_input))
        )
    ]

    # zero treatment
    if (transformation_table[:, 1] == 0).any():
        transformation_table[:, 2][transformation_table[:, 1] == 0] = np.median(
            transformation_table[:, 2][transformation_table[:, 1] == 0]
        )

    # sort back
    transformation_table = transformation_table[
                           np.unique(transformation_table[:, 1], return_index=True)[1], :
                           ]
    transformation_table = transformation_table[transformation_table[:, 0].argsort()]

    # create back-transformation interpolator
    back_transform = interpolate.interp1d(
        transformation_table[:, 2],
        transformation_table[:, 1],
        kind="linear",
        fill_value=(0, data_input.max() + data_input.max() * 0.10),
        bounds_error=False,
    )
    # create transformation interpolator
    transform = interpolate.interp1d(
        transformation_table[:, 1],
        transformation_table[:, 2],
        kind="linear",
        fill_value=(0, data_input.max()),
        bounds_error=False,
    )

    return transform, back_transform
