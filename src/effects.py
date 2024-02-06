import numpy as np

wah_wah_damp = 0.1
wah_wah_min_freq = 500
wah_wah_max_freq = 5000
wah_wah_wah_freq = 2000


def norm_signal(input_signal):
    output_signal = input_signal / np.max(np.absolute(input_signal))
    return output_signal


def apply_wah_wah(input_signal, sample_rate, damp=wah_wah_damp, min_freq=wah_wah_min_freq, max_freq=wah_wah_max_freq, wah_freq=wah_wah_wah_freq):
    output_signal = np.zeros(len(input_signal))

    outh = np.zeros(len(input_signal))
    outl = np.zeros(len(input_signal))

    delta = wah_freq / sample_rate
    centerf = np.concatenate((np.arange(min_freq, max_freq, delta), np.arange(max_freq, min_freq, -delta)))

    while len(centerf) < len(input_signal):
        centerf = np.concatenate((centerf, centerf))

    centerf = centerf[:len(input_signal)]

    f1 = 2 * np.sin(np.pi * centerf[0] / sample_rate)
    outh[0] = input_signal[0]
    output_signal[0] = f1 * outh[0]
    outl[0] = f1 * output_signal[0]

    for n in range(1, len(input_signal)):
        outh[n] = input_signal[n] - outl[n-1] -  2 * damp * output_signal[n-1]
        output_signal[n] = f1 * outh[n] + output_signal[n-1]
        outl[n] = f1 * output_signal[n] + outl[n-1]
        f1 = 2 * np.sin(np.pi * centerf[n] / sample_rate)

    output_signal = norm_signal(output_signal)

    return output_signal, sample_rate
