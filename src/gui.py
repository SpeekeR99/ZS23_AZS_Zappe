from utils import to_stereo
import numpy as np
from OpenGL.GL import *
import glfw
import imgui
import scipy.io.wavfile
import sounddevice as sd
import matplotlib.pyplot as plt
import librosa
import librosa.display

# Window name
WINDOW_NAME = "Sound Effects Semester Work"
#  Window width
WINDOW_WIDTH = 1280
#  Window height
WINDOW_HEIGHT = 720

#  Whether to show the settings window
show_settings_window = False
#  Whether to show the about window
show_about_window = False
#  Whether to show the save as dialog window
show_save_as_dialog = False

#  Loaded sounds
sounds = {}
#  Currently selected sound
current_sound = 0

#  Different sound effects
effect_names = ["Ping Pong Delay", "Wah Wah", "Flanger", "Phaser", "Overdrive", "Distortion", "Reverb", "Bitcrusher", "8D Audio", "Vocal Doubler"]
#  Currently selected effect
current_effect = 0

#  Whether to show the effects window
show_effects_window = True


def load_sound(filepath):
    """
    Loads a sound from the given filepath
    :param filepath: Filepath of the sound
    :return: None
    """
    global sounds
    filepath = filepath.replace("\\", "/")
    sample_rate, audio = scipy.io.wavfile.read(filepath)
    if audio is None:
        print("Error loading sound: " + filepath + "!")
        return
    if audio.ndim == 1:
        second_channel = np.copy(audio)
        audio = to_stereo(audio, second_channel)
    name = avoid_name_duplicates(filepath)
    sounds[name] = {"data": audio, "sample_rate": sample_rate, "show": True}


def avoid_name_duplicates(filepath):
    """
    Checks if the sound name already exists in the global dictionary of sounds
    If it does, it adds a number to the end of the name
    :param filepath: Filepath of the sound
    :return: Unique name of the sound
    """
    name = filepath.split("/")[-1].split(".")[0]
    extension = filepath.split("/")[-1].split(".")[-1]
    copy = 1
    while name + "." + extension in sounds:
        if copy == 1:
            name = name + " (1)"
        else:
            name = name[:-3]
            name = name + "(" + str(copy) + ")"
        copy += 1
    return name + "." + extension


def play_sound(name):
    sound = sounds[name]
    audio = np.int16(sound["data"]/np.max(np.abs(sound["data"])) * 32767)
    sample_rate = sound["sample_rate"]
    sd.play(audio, sample_rate)


def show_sound(name):
    """
    Creates an ImGui window for the sound
    :param name: Name of the sound
    :return: Boolean whether to close the window or not
    """
    global sounds
    imgui.set_next_window_size(600, 320)
    imgui.set_next_window_position(50 + np.random.randint(-25, 25), 50 + np.random.randint(-25, 25), imgui.FIRST_USE_EVER)
    _, close_bool = imgui.begin(name, True, imgui.WINDOW_NO_SAVED_SETTINGS | imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_RESIZE)

    imgui.text("Sample rate: " + str(sounds[name]["sample_rate"]) + " Hz")
    imgui.text("Length: " + str(round(len(sounds[name]["data"]) / sounds[name]["sample_rate"] * 100) / 100) + " s")

    plot_left_channel = sounds[name]["data"][:, 0].astype(np.float32)
    left_min, left_max = np.min(plot_left_channel), np.max(plot_left_channel)
    plot_right_channel = sounds[name]["data"][:, 1].astype(np.float32)
    right_min, right_max = np.min(plot_right_channel), np.max(plot_right_channel)

    imgui.plot_lines("L", plot_left_channel, scale_min=left_min, scale_max=left_max, graph_size=(600, 100), overlay_text="Left channel")
    imgui.plot_lines("R", plot_right_channel, scale_min=right_min, scale_max=right_max, graph_size=(600, 100), overlay_text="Right channel")

    if imgui.button("Play"):
        play_sound(name)
    imgui.same_line()
    if imgui.button("Stop"):
        sd.stop()

    if imgui.button("Spectrogram"):
        plot_spectrogram(name)

    imgui.end()
    return close_bool


def impl_glfw_init():
    """
    Initialize glfw and return the window
    :return: Window object
    """
    if not glfw.init():
        print("Could not initialize OpenGL context")
        exit(1)

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
    glfw.window_hint(glfw.RESIZABLE, GL_FALSE)

    window = glfw.create_window(int(WINDOW_WIDTH), int(WINDOW_HEIGHT), WINDOW_NAME, None, None)
    glfw.make_context_current(window)
    mode = glfw.get_video_mode(glfw.get_primary_monitor())
    glfw.set_window_pos(window, int((mode.size.width - WINDOW_WIDTH) / 2), int((mode.size.height - WINDOW_HEIGHT) / 2))

    if not window:
        glfw.terminate()
        print("Could not initialize Window")
        exit(1)

    return window


def my_text_separator(text):
    """
    Creates a separator with text in the middle
    :param text: Text to display
    :return: None
    """
    imgui.separator()
    imgui.text(text)
    imgui.separator()


def plot_spectrogram(name):
    audio_stereo = sounds[name]["data"]
    audio_mono = np.mean(audio_stereo, axis=1).astype(np.float32)
    sample_rate = sounds[name]["sample_rate"]

    plt.figure(figsize=(12, 6))
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(audio_mono)), ref=np.max),
                             y_axis="log", x_axis="time", sr=sample_rate, cmap="plasma")
    plt.colorbar(format="%+2.0f dB")
    plt.title(name)
    plt.show(block=False)
