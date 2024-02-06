import numpy as np


def extract_channels(signal):
    """
    Extracts the left and right channels from a stereo signal
    :param signal: Input stereo signal
    :return: Left and right channels
    """
    return signal[:, 0], signal[:, 1]


def to_stereo(left_channel, right_channel):
    """
    Combines two mono channels into a stereo signal
    :param left_channel: Left channel
    :param right_channel: Right channel
    :return: Stereo signal
    """
    return np.vstack((left_channel, right_channel)).T
