'''
based on "CustomFactorExample.py" from gtsam
by Author: Fan Jiang, Frank Dellaert
'''

from functools import partial
from typing import List, Optional

import gtsam
import numpy as np


def error_odom(measurement: np.ndarray, this: gtsam.CustomFactor,
               values: gtsam.Values,
               jacobians: Optional[List[np.ndarray]]) -> float:
    """Odometry Factor error function
    :param measurement: Odometry measurement, to be filled with `partial`
    :param this: gtsam.CustomFactor handle
    :param values: gtsam.Values
    :param jacobians: Optional list of Jacobians
    :return: the unwhitened error
    """
    key1 = this.keys()[0]
    key2 = this.keys()[1]
    pos1, pos2 = values.atVector(key1), values.atVector(key2)
    error = measurement - (pos1 - pos2)
    if jacobians is not None:
        jacobians[0] = I
        jacobians[1] = -I

    return error
