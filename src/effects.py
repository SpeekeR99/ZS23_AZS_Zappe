import numpy as np

wah_wah_damp = 0.1
wah_wah_min_freq = 500
wah_wah_max_freq = 5000
wah_wah_wah_freq = 2000

flanger_max_delay = 12
flanger_freq = 2
flanger_gain = 0.6


def norm_signal(input_signal):
    output_signal = input_signal / np.max(np.absolute(input_signal))
    return output_signal


def apply_wah_wah(input_signal, sample_rate):
    output_signal = np.zeros(len(input_signal))

    outh = np.zeros(len(input_signal))
    outl = np.zeros(len(input_signal))

    delta = wah_wah_wah_freq / sample_rate
    centerf = np.concatenate((np.arange(wah_wah_min_freq, wah_wah_max_freq, delta), np.arange(wah_wah_max_freq, wah_wah_min_freq, -delta)))

    while len(centerf) < len(input_signal):
        centerf = np.concatenate((centerf, centerf))

    centerf = centerf[:len(input_signal)]

    f1 = 2 * np.sin(np.pi * centerf[0] / sample_rate)
    outh[0] = input_signal[0]
    output_signal[0] = f1 * outh[0]
    outl[0] = f1 * output_signal[0]

    for n in range(1, len(input_signal)):
        outh[n] = input_signal[n] - outl[n-1] - 2 * wah_wah_damp * output_signal[n-1]
        output_signal[n] = f1 * outh[n] + output_signal[n-1]
        outl[n] = f1 * output_signal[n] + outl[n-1]
        f1 = 2 * np.sin(np.pi * centerf[n] / sample_rate)

    output_signal = norm_signal(output_signal)

    return output_signal, sample_rate


def apply_flanger(input_signal, sample_rate):
    num = int(flanger_max_delay * 1e-3 * sample_rate
              )
    output_signal = np.zeros(len(input_signal))

    for n in range(len(input_signal)):
        d = int(0.5 * num * (1 + np.sin(2 * np.pi * flanger_freq * n / sample_rate)))
        if d < n:
            output_signal[n] = input_signal[n] + flanger_gain * input_signal[n-d]
        else:
            output_signal[n] = input_signal[n]

    output_signal = norm_signal(output_signal)

    return output_signal, sample_rate
