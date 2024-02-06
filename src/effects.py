import numpy as np
from scipy.signal import fftconvolve
import scipy.io.wavfile

# Ping pong delay parameters
ping_pong_delay_time = 0.1
ping_pong_feedback = 0.5
ping_pong_mix = 0.5

# Wah wah parameters
wah_wah_damp = 0.1
wah_wah_min_freq = 500
wah_wah_max_freq = 5000
wah_wah_wah_freq = 2000

# Flanger parameters
flanger_freq = 0.1
flanger_depth = 0.002
flanger_mix = 0.5

# Phaser parameters
phaser_max_delay = 20
phaser_freq = 0.5
phaser_gain = 0.6

# Overdrive parameters
overdrive_threshold = 0.2

# Distortion parameters
distortion_gain = 15

# Reverb parameters
reverb_ir_names = ["Church", "Forest", "Cave", "Space"]
reverb_ir_filepaths = ["../sounds/impulse_responses/ir_church.wav", "../sounds/impulse_responses/ir_forest.wav",
                       "../sounds/impulse_responses/ir_cave.wav", "../sounds/impulse_responses/ir_space.wav"]
reverb_ir_index = 0
reverb_mix = 0.5

# Bitcrusher parameters
bit_crusher_bits = 4

# 8D audio parameters
eight_d_spin_speed = 0.1

# Vocal doubler parameters
vocal_doubler_delay = 1
vocal_doubler_detune_cents = 10


def norm_signal(input_signal):
    """
    Normalizes the input signal
    :param input_signal: Input signal
    :return: Normalized signal
    """
    output_signal = input_signal / np.max(np.absolute(input_signal))
    return output_signal


def apply_ping_pong_delay(input_signal, sample_rate):
    """
    Applies a ping pong delay effect to the input signal
    :param input_signal: Input signal
    :param sample_rate: Sample rate of the input signal
    :return: Output signal and it's sample rate
    """
    num_samples = len(input_signal)
    delay_samples = int(ping_pong_delay_time * sample_rate)
    delayed_stereo = np.zeros((num_samples, 2))

    # Apply the ping pong delay effect
    for i in range(delay_samples, num_samples):
        delayed_stereo[i, 0] = input_signal[i, 0] + ping_pong_feedback * input_signal[i - delay_samples, 1]
        delayed_stereo[i, 1] = input_signal[i, 1] + ping_pong_feedback * input_signal[i - delay_samples, 0]

    # Mix the input signal with the delayed signal
    out_audio = ping_pong_mix * input_signal + (1 - ping_pong_mix) * delayed_stereo

    return out_audio, sample_rate


def apply_wah_wah(input_signal, sample_rate):
    """
    Applies a wah wah effect to the input signal
    :param input_signal: Input signal
    :param sample_rate: Sample rate of the input signal
    :return: Output signal and it's sample rate
    """
    output_signal = np.zeros(len(input_signal))

    # High pass and low pass filters
    outh = np.zeros(len(input_signal))
    outl = np.zeros(len(input_signal))

    # Create the center frequency vector
    delta = wah_wah_wah_freq / sample_rate
    centerf = np.concatenate((np.arange(wah_wah_min_freq, wah_wah_max_freq, delta), np.arange(wah_wah_max_freq, wah_wah_min_freq, -delta)))

    # Repeat the center frequency vector to match the length of the input signal
    while len(centerf) < len(input_signal):
        centerf = np.concatenate((centerf, centerf))

    # Truncate the center frequency vector to match the length of the input signal
    centerf = centerf[:len(input_signal)]

    # Initialize the first sample
    f1 = 2 * np.sin(np.pi * centerf[0] / sample_rate)
    outh[0] = input_signal[0]
    output_signal[0] = f1 * outh[0]
    outl[0] = f1 * output_signal[0]

    # Apply the wah wah effect
    for n in range(1, len(input_signal)):
        outh[n] = input_signal[n] - outl[n-1] - 2 * wah_wah_damp * output_signal[n-1]
        output_signal[n] = f1 * outh[n] + output_signal[n-1]
        outl[n] = f1 * output_signal[n] + outl[n-1]
        f1 = 2 * np.sin(np.pi * centerf[n] / sample_rate)

    return output_signal, sample_rate


def apply_flanger(input_signal, sample_rate):
    """
    Applies a flanger effect to the input signal
    :param input_signal: Input signal
    :param sample_rate: Sample rate of the input signal
    :return: Output signal and it's sample rate
    """
    output_signal = np.zeros_like(input_signal, dtype=np.float32)

    # Create the low frequency oscillator
    time = np.arange(len(input_signal)) / sample_rate
    lfo = np.sin(2 * np.pi * flanger_freq * time) * flanger_depth

    # Apply the flanger effect
    for i in range(len(input_signal)):
        delay = int(lfo[i] * sample_rate)
        if i + delay < len(input_signal):
            output_signal[i] = flanger_mix * input_signal[i] + (1 - flanger_mix) * input_signal[i + delay]

    return output_signal, sample_rate


def apply_phaser(input_signal, sample_rate):
    """
    Applies a phaser effect to the input signal
    :param input_signal: Input signal
    :param sample_rate: Sample rate of the input signal
    :return: Output signal and it's sample rate
    """
    num = int(phaser_max_delay * 1e-3 * sample_rate)
    output_signal = np.zeros(len(input_signal))

    # Apply the phaser effect
    for n in range(len(input_signal)):
        # Calculate the delay
        d = int(0.5 * num * (1 + np.sin(2 * np.pi * phaser_freq * n / sample_rate)))
        if d < n:  # Avoid negative indices
            output_signal[n] = input_signal[n] + phaser_gain * input_signal[n - d]
        else:
            output_signal[n] = input_signal[n]

    return output_signal, sample_rate


def apply_overdrive(input_signal, sample_rate):
    """
    Applies an overdrive effect to the input signal
    :param input_signal: Input signal
    :param sample_rate: Sample rate of the input signal
    :return: Output signal and it's sample rate
    """
    output_signal = np.zeros(len(input_signal))

    # Apply the overdrive effect
    # Basically a for loop with three if statements
    # For each sample, check if it's less than the overdrive threshold
    output_signal = np.where(np.absolute(input_signal) < overdrive_threshold, 2 * input_signal, output_signal)
    # If it's greater than the overdrive threshold, apply the overdrive function
    output_signal = np.where(np.absolute(input_signal) >= overdrive_threshold, np.where(input_signal > 0, (3 - (2 - 3 * input_signal) ** 2) / 3, -(3 - (2 - 3 * np.absolute(input_signal)) ** 2) / 3), output_signal)
    # If it's greater than 2 times the overdrive threshold, set it to 1 or -1
    output_signal = np.where(np.absolute(input_signal) > 2 * overdrive_threshold, np.where(input_signal > 0, 1, -1), output_signal)

    return output_signal, sample_rate


def apply_distortion(input_signal, sample_rate):
    """
    Applies a distortion effect to the input signal
    :param input_signal: Input signal
    :param sample_rate: Sample rate of the input signal
    :return: Output signal and it's sample rate
    """
    q = np.sign(input_signal)

    # Alpha is the distortion gain (negative)
    alpha = -1 * float(distortion_gain)
    # Apply the distortion effect
    output_signal = q * (1 - np.exp(alpha * q * input_signal))

    return output_signal, sample_rate


def apply_reverb(input_signal, sample_rate):
    """
    Applies a reverb effect to the input signal
    :param input_signal: Input signal
    :param sample_rate: Sample rate of the input signal
    :return: Output signal and it's sample rate
    """
    # Pick the chosen impulse response
    ir_filepath = reverb_ir_filepaths[reverb_ir_index]
    _, impulse_response = scipy.io.wavfile.read(ir_filepath)
    if impulse_response.ndim > 1:  # If the impulse response is stereo, convert it to mono
        impulse_response = impulse_response[:, 0]

    # Normalize the impulse response
    norm_signal(impulse_response)

    # Apply the reverb effect == convolve the input signal with the impulse response
    output_signal = fftconvolve(input_signal, impulse_response, mode="full")
    output_signal = reverb_mix * input_signal + (1 - reverb_mix) * output_signal[:len(input_signal)]
    return output_signal, sample_rate


def apply_bitcrusher(input_signal, sample_rate):
    """
    Applies a bit crusher effect to the input signal
    :param input_signal: Input signal
    :param sample_rate: Sample rate of the input signal
    :return: Output signal and it's sample rate
    """
    # Calculate the maximum value for the bit crusher
    max_value = 2 ** (bit_crusher_bits - 1) - 1

    # Apply the bit crusher effect
    output_signal = np.round(input_signal * max_value) / max_value

    return output_signal, sample_rate


def apply_8d_audio(input_signal, sample_rate):
    """
    Applies an 8D audio effect to the input signal
    :param input_signal: Input signal
    :param sample_rate: Sample rate of the input signal
    :return: Output signal and it's sample rate
    """
    # Initialize the output signal
    temp_audio = np.zeros((len(input_signal), 4), dtype=np.float32)
    out_audio = np.zeros((len(input_signal), 2), dtype=np.float32)
    # Convert the input signal to mono
    input_signal_mono = np.mean(input_signal, axis=1)

    # Apply the 8D audio effect
    for i in range(len(input_signal_mono)):
        # Calculate the pan for the left and right channels
        pan_l_r = np.sin(2 * np.pi * (i / sample_rate * eight_d_spin_speed))
        # Left and right gain
        left_gain = np.cos(pan_l_r * 0.5 * np.pi) + 0.25
        right_gain = np.sin(pan_l_r * 0.5 * np.pi) + 0.25

        # Calculate the pan for the front and back channels
        pan_f_b = np.sin(2 * np.pi * (i / sample_rate * eight_d_spin_speed))
        # Front and back gain
        front_gain = np.abs(np.cos(pan_f_b * 0.5 * np.pi)) + 0.1
        back_gain = np.abs(np.sin(pan_f_b * 0.5 * np.pi)) + 0.1

        # Multiply the input signal by the gains and create 4 channels (directions)
        temp_audio[i, 0] = (left_gain * front_gain) * input_signal_mono[i]
        temp_audio[i, 1] = (right_gain * front_gain) * input_signal_mono[i]
        temp_audio[i, 2] = (left_gain * back_gain) * input_signal_mono[i]
        temp_audio[i, 3] = (right_gain * back_gain) * input_signal_mono[i]

        # Sum the channels to get the output signal (stereo)
        out_audio[i, 0] = temp_audio[i, 0] + temp_audio[i, 2]
        out_audio[i, 1] = temp_audio[i, 1] + temp_audio[i, 3]

    return out_audio, sample_rate


def apply_vocal_doubler(input_signal, sample_rate):
    """
    Applies a vocal doubler effect to the input signal
    :param input_signal: Input signal
    :param sample_rate: Sample rate of the input signal
    :return: Output signal and it's sample rate
    """
    input_copy = input_signal.copy()
    delay_samples = int(vocal_doubler_delay * sample_rate / 1000)

    # Create a detuned duplicate of the input signal
    duplicate_audio = np.concatenate((np.zeros(delay_samples), input_copy[:-delay_samples]))
    # Calculate the pitch shift factor
    pitch_shift_factor = 2 ** (vocal_doubler_detune_cents / 1200)
    # Interpolate the duplicate audio to create the detuned duplicate
    detuned_duplicate = np.interp(np.arange(len(duplicate_audio)) / pitch_shift_factor, np.arange(len(duplicate_audio)), duplicate_audio)

    # Mix the input signal with the detuned duplicate
    output_signal = 0.5 * input_copy + 0.5 * detuned_duplicate

    return output_signal, sample_rate
