# -*- coding: utf-8 -*-

import numpy as np


def number_to_integral(number):
    return int(np.ceil(number))


def threshold_color_array(src):
    return np.maximum(np.minimum(src, 255), 0).astype(np.uint8)
