from utils import *
import numpy as np

ping_pong_delay_time = 0.1
ping_pong_feedback = 0.5
ping_pong_mix = 0.5

wah_wah_damp = 0.1
wah_wah_min_freq = 500
wah_wah_max_freq = 5000
wah_wah_wah_freq = 2000

flanger_max_delay = 12
flanger_freq = 2
flanger_gain = 0.6

# phaser

overdrive_threshold = 0.2

distortion_gain = 15

# reverb

bit_crusher_bits = 4

eight_d_spin_speed = 0.1

vocal_doubler_delay = 1
vocal_doubler_detune_cents = 10


def norm_signal(input_signal):
    output_signal = input_signal / np.max(np.absolute(input_signal))
    return output_signal


def apply_ping_pong_delay(input_signal, sample_rate):
    num_samples = len(input_signal)
    delay_samples = int(ping_pong_delay_time * sample_rate)
    delayed_stereo = np.zeros((num_samples, 2))

    for i in range(delay_samples, num_samples):
        delayed_stereo[i, 0] = input_signal[i, 0] + ping_pong_feedback * input_signal[i - delay_samples, 1]
        delayed_stereo[i, 1] = input_signal[i, 1] + ping_pong_feedback * input_signal[i - delay_samples, 0]

    out_audio = ping_pong_mix * input_signal + (1 - ping_pong_mix) * delayed_stereo

    return out_audio, sample_rate


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

    return output_signal, sample_rate


def apply_flanger(input_signal, sample_rate):
    num = int(flanger_max_delay * 1e-3 * sample_rate)
    output_signal = np.zeros(len(input_signal))

    for n in range(len(input_signal)):
        d = int(0.5 * num * (1 + np.sin(2 * np.pi * flanger_freq * n / sample_rate)))
        if d < n:
            output_signal[n] = input_signal[n] + flanger_gain * input_signal[n-d]
        else:
            output_signal[n] = input_signal[n]

    return output_signal, sample_rate


def apply_phaser(input_signal, sample_rate):
    return input_signal, sample_rate


def apply_overdrive(input_signal, sample_rate):
    output_signal = np.zeros(len(input_signal))

    output_signal = np.where(np.absolute(input_signal) < overdrive_threshold, 2 * input_signal, output_signal)
    output_signal = np.where(np.absolute(input_signal) >= overdrive_threshold, np.where(input_signal > 0, (3 - (2 - 3 * input_signal) ** 2) / 3, -(3 - (2 - 3 * np.absolute(input_signal)) ** 2) / 3), output_signal)
    output_signal = np.where(np.absolute(input_signal) > 2 * overdrive_threshold, np.where(input_signal > 0, 1, -1), output_signal)

    return output_signal, sample_rate


def apply_distortion(input_signal, sample_rate):
    q = np.sign(input_signal)

    alpha = -1 * float(distortion_gain)
    output_signal = q * (1 - np.exp(alpha * q * input_signal))

    return output_signal, sample_rate


def apply_reverb(input_signal, sample_rate):
    return input_signal, sample_rate


def apply_bitcrusher(input_signal, sample_rate):
    max_value = 2 ** (bit_crusher_bits - 1) - 1

    output_signal = np.round(input_signal * max_value) / max_value

    return output_signal, sample_rate


def apply_8d_audio(input_signal, sample_rate):
    temp_audio = np.zeros((len(input_signal), 4), dtype=np.float32)
    out_audio = np.zeros((len(input_signal), 2), dtype=np.float32)
    input_signal_mono = np.mean(input_signal, axis=1)

    for i in range(len(input_signal_mono)):
        pan_l_r = np.sin(2 * np.pi * (i / sample_rate * eight_d_spin_speed))
        left_gain = np.cos(pan_l_r * 0.5 * np.pi) + 0.25
        right_gain = np.sin(pan_l_r * 0.5 * np.pi) + 0.25

        pan_f_b = np.sin(2 * np.pi * (i / sample_rate * eight_d_spin_speed))
        front_gain = np.abs(np.cos(pan_f_b * 0.5 * np.pi)) + 0.1
        back_gain = np.abs(np.sin(pan_f_b * 0.5 * np.pi)) + 0.1

        temp_audio[i, 0] = (left_gain * front_gain) * input_signal_mono[i]
        temp_audio[i, 1] = (right_gain * front_gain) * input_signal_mono[i]
        temp_audio[i, 2] = (left_gain * back_gain) * input_signal_mono[i]
        temp_audio[i, 3] = (right_gain * back_gain) * input_signal_mono[i]

        out_audio[i, 0] = temp_audio[i, 0] + temp_audio[i, 2]
        out_audio[i, 1] = temp_audio[i, 1] + temp_audio[i, 3]

    return out_audio, sample_rate


def apply_vocal_doubler(input_signal, sample_rate):
    input_copy = input_signal.copy()
    delay_samples = int(vocal_doubler_delay * sample_rate / 1000)

    duplicate_audio = np.concatenate((np.zeros(delay_samples), input_copy[:-delay_samples]))
    pitch_shift_factor = 2 ** (vocal_doubler_detune_cents / 1200)
    detuned_duplicate = np.interp(np.arange(len(duplicate_audio)) / pitch_shift_factor, np.arange(len(duplicate_audio)), duplicate_audio)

    output_signal = 0.5 * input_copy + 0.5 * detuned_duplicate

    return output_signal, sample_rate
