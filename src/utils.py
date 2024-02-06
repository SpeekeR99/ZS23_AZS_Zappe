import numpy as np


def extract_channels(signal):
    return signal[:, 0], signal[:, 1]


def to_stereo(left_channel, right_channel):
    return np.vstack((left_channel, right_channel)).T
