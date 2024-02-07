# Audio Effects Implementation

This project implements various audio effects using digital signal processing techniques in Python.

## Implemented Effects

The following audio effects are implemented:

*   Ping Pong Delay
*   Wah-Wah
*   Flanger
*   Phaser
*   Overdrive
*   Distortion
*   Reverb
*   Bit Crusher
*   8D Audio
*   Vocal Doubler

Most effects are more suitable for musical instruments than for the human voice, with the exception of the Vocal Doubler effect.

## Implementation Details

The audio effects are implemented in `Python 3.9`, primarily using the `NumPy` and `SciPy` libraries for digital signal processing.
The implementation includes a graphical user interface (GUI) created using `imgui` and `wxWidgets` with the `OpenGL API`.

### Effect Implementations:

*   **Ping Pong Delay:**  A delayed signal is calculated for both left and right channels. The delayed sample from the right channel is added to each sample of the left channel, and vice versa. The input signal is then mixed with the delayed signal.
*   **Wah-Wah:**  An array of center frequencies is created, and the effect is applied to the input signal using filters and a sinusoidal modulation function.
*   **Flanger:** A low-frequency oscillator is created and used to modulate the signal delay. The Flanger effect is then applied to the input signal using this oscillator.
*   **Phaser:**  The maximum number of delay samples is calculated based on the maximum delay and the sampling rate. The effect is applied by calculating the delay and modulating the signal amplitude based on the selected frequency.
*   **Overdrive:**  The signal is amplified based on a defined threshold, and a more complex calculation is performed to determine the output signal value.
*   **Distortion:** The sign of the input signal is taken, and a distortion coefficient is calculated from the specified gain parameter.  The distortion effect is then applied to the input signal using an exponential function.
*   **Reverb:** Implemented using convolution of the input signal with an impulse response. The resulting signal is a mix of the original input with the result of the convolution.  Impulse responses were downloaded from the internet (a choice of four is available in the GUI).  The church impulse response is from [https://afewthingz.com/impulseresponse](https://afewthingz.com/impulseresponse). Other impulse responses are from [https://pixabay.com/sound-effects](https://pixabay.com/sound-effects).
*   **Bit Crusher:** Implemented as a simple quantization. Values of the input signal samples are rounded to the nearest integer and then divided by the maximum value.
*   **8D Audio:** The input signal is converted to mono. Panoramic gains are calculated for the front, back, left, and right channels. The input signal is multiplied by these gains and distributed to the four channels.  Finally, these channels are summed to obtain the output signal (stereo).
*   **Vocal Doubler:** A copy of the input signal is created, and the number of delay samples is calculated. A duplicate of the input signal with a time shift and detuning (pitch shift) is created using interpolation.  Both signals (original and detuned) are then mixed together in a 1:1 ratio.

## User Guide

1.  **Install Dependencies:**  From the project's root directory, use the following command to install the required Python packages:

    ```
    pip install -r requirements.txt
    ```

2.  **Run the Application:**  From the `/src/` directory, run the following command:

    ```
    python main.py
    ```

3.  **GUI Overview:**  The main application window provides the following menu options:

    *   **File:** Reset workspace, load sound, save sound, and exit application.
    *   **Audio Effects:** Open the audio effects pane (this pane opens automatically at startup but can be reopened here).
    *   **Settings:** Window settings (window size and color scheme).
    *   **Help:** About (displays the author's name).

4.  **Sound Information Windows:** Each time a new sound is loaded or generated using an effect, a new window opens containing information about the sound (sampling frequency, sound length, graph of the left and right channels (amplitude vs. time)), playback controls (play, stop), and the option to display a spectrogram.

5.  **Audio Effects Pane:** The audio effects pane has:

    *   A dropdown list for selecting the audio to which the effect should be applied.
    *   A dropdown list for selecting the desired audio effect.
    *   A modular interface that changes based on the selected effect, allowing for specific parameter adjustments.
    *   An "Apply effect" button to apply the selected effect with the chosen parameters.

## Results

The results can be evaluated by listening to the processed audio.
The GUI also displays the amplitude vs. time of each loaded sound and allows the user to view the spectrogram (frequency vs. time).

## Requirements

*   Python 3.9
*   NumPy
*   SciPy
*   imgui
*   wxWidgets
*   OpenGL

A `requirements.txt` file is provided to ease the installation of the dependencies.
