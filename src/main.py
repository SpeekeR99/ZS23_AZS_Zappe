from gui import *
import effects
from effects import *
from utils import *
from imgui.integrations.glfw import GlfwRenderer
import wx


def apply_effect_button_callback():
    """
    Callback for the apply effect button
    Applies the currently selected effect with the currently selected parameters to the currently selected sound
    """
    # Array of effect functions
    effect_functions = [apply_ping_pong_delay, apply_wah_wah, apply_flanger, apply_phaser, apply_overdrive, apply_distortion, apply_reverb,
                        apply_bitcrusher, apply_8d_audio, apply_vocal_doubler]

    # Get the current sound
    name = list(sounds.keys())[current_sound]
    sound = sounds[name]

    audio = sound["data"]
    audio_l, audio_r = extract_channels(audio)
    audio_l = norm_signal(audio_l)
    audio_r = norm_signal(audio_r)

    sample_rate = sound["sample_rate"]

    # Get the current effect
    effect_func = effect_functions[current_effect]
    # Apply the effect
    if effect_func == apply_8d_audio or effect_func == apply_ping_pong_delay:  # These effects require stereo audio
        out_audio, out_sample_rate = effect_func(audio, sample_rate)
        out_audio_l, out_audio_r = extract_channels(out_audio)

    else:  # All other work for both mono and stereo audio
        out_audio_l, out_sample_rate = effect_func(audio_l, sample_rate)
        out_audio_r, out_sample_rate = effect_func(audio_r, sample_rate)

    # Normalize the output
    out_audio_l = norm_signal(out_audio_l)
    out_audio_r = norm_signal(out_audio_r)

    # Convert the output to stereo
    out_audio = to_stereo(out_audio_l, out_audio_r)

    # Add the new sound to the dictionary
    new_name = avoid_name_duplicates(name.split(".")[0] + " (" + effect_names[current_effect] + ")." + name.split(".")[-1])
    sounds[new_name] = {"data": out_audio, "sample_rate": out_sample_rate, "show": True}


def main():
    """
    Main function
    """
    global WINDOW_WIDTH, WINDOW_HEIGHT, show_settings_window, show_about_window, show_save_as_dialog, sounds, \
        current_sound, effect_names, current_effect, show_effects_window
    # Initialize the GUI
    app = wx.App()
    app.MainLoop()
    imgui.create_context()
    imgui.style_colors_dark()
    window = impl_glfw_init()
    impl = GlfwRenderer(window)
    mode = glfw.get_video_mode(glfw.get_primary_monitor())

    to_be_deleted = None
    current_style = 0
    background_color = (29. / 255, 29. / 255, 29. / 255)

    # Main loop
    while not glfw.window_should_close(window):
        glfw.poll_events()
        impl.process_inputs()

        imgui.new_frame()

        # Menu bar
        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("File"):
                clicked_new, _ = imgui.menu_item("New Blank Project", None, False, True)
                if clicked_new:
                    sure = wx.MessageDialog(
                        None,
                        "Are you sure you want to create a new project? All unsaved changes will be lost!",
                        "New Project", wx.YES_NO | wx.ICON_QUESTION
                    ).ShowModal()
                    if sure == wx.ID_YES:
                        sounds = {}
                        show_about_window = False
                        show_settings_window = False

                clicked_load, _ = imgui.menu_item("Load sound...", None, False, True)
                if clicked_load:
                    filepath = wx.FileSelector(
                        "Load sound",
                        wildcard="Sound Files (*.wav)|*.wav"
                    )
                    if filepath:
                        load_sound(filepath)
                        pass

                clicked_save, _ = imgui.menu_item("Save sound as...", None, False, True)
                if clicked_save:
                    show_save_as_dialog = True

                imgui.separator()

                clicked_exit, _ = imgui.menu_item("Exit", 'Alt+F4', False, True)
                if clicked_exit:
                    glfw.set_window_should_close(window, True)

                imgui.end_menu()
            if imgui.begin_menu("Audio effects"):
                clicked_choose, _ = imgui.menu_item("Choose effect...", None, False, True)
                if clicked_choose:
                    show_effects_window = True

                imgui.end_menu()
            if imgui.begin_menu("Settings"):

                clicked_settings, _ = imgui.menu_item("Window Settings...", None, False, True)
                if clicked_settings:
                    show_settings_window = True

                imgui.end_menu()
            if imgui.begin_menu("Help"):

                clicked_about, _ = imgui.menu_item("About...", None, False, True)
                if clicked_about:
                    show_about_window = True

                imgui.end_menu()
            imgui.end_main_menu_bar()

        # Closed effects
        if to_be_deleted:
            sounds.pop(to_be_deleted)
            to_be_deleted = None

        for name in sounds:
            if sounds[name]["show"]:
                sounds[name]["show"] = show_sound(name)
                if not sounds[name]["show"]:
                    sure = wx.MessageDialog(
                        None,
                        "Are you sure you want to delete " + name + "? All unsaved changes will be lost!",
                        "Delete sound", wx.YES_NO | wx.ICON_QUESTION
                    ).ShowModal()
                    if sure == wx.ID_YES:
                        to_be_deleted = name
                    else:
                        sounds[name]["show"] = True

        # Save as dialog
        if show_save_as_dialog:
            imgui.set_next_window_size(500, 100, imgui.ONCE)
            imgui.set_next_window_position((WINDOW_WIDTH - 500) / 2, (WINDOW_HEIGHT - 100) / 2, imgui.ONCE)

            _, show_save_as_dialog = imgui.begin("Save sound as...", True, imgui.WINDOW_NO_COLLAPSE)

            imgui.text("Sound Selection:")
            _, current_sound = imgui.combo("Sound", current_sound, list(sounds.keys()))

            if imgui.button("Save as..."):
                if len(list(sounds.keys())) == 0 or current_sound > len(list(sounds.keys())):
                    print("No sound selected!")
                else:
                    filepath = wx.FileSelector(
                        "Save sound as...", default_filename=list(sounds.keys())[current_sound],
                        wildcard="Sound Files (*.wav)|*.wav",
                        flags=wx.FD_SAVE
                    )
                    if filepath:
                        sound = sounds[list(sounds.keys())[current_sound]]
                        scipy.io.wavfile.write(filepath, sound["sample_rate"], sound["data"])

            imgui.end()

        # Effects window
        if show_effects_window:
            imgui.set_next_window_size(500, 500, imgui.ONCE)
            imgui.set_next_window_position((WINDOW_WIDTH - 500) / 2 + 300, (WINDOW_HEIGHT - 500) / 2, imgui.ONCE)

            _, show_effects_window = imgui.begin("Audio effects", True, imgui.WINDOW_NO_COLLAPSE)

            my_text_separator("Sound selection")
            _, current_sound = imgui.combo("Sound", current_sound, list(sounds.keys()))

            my_text_separator("Effect selection")
            _, current_effect = imgui.combo("Effect", current_effect, effect_names)

            if current_effect == 0:  # Ping pong delay
                my_text_separator("Ping-pong delay settings")

                _, effects.ping_pong_delay_time = imgui.slider_float("Delay time (s)", effects.ping_pong_delay_time, 0.0, 1.0)
                _, effects.ping_pong_feedback = imgui.slider_float("Feedback", effects.ping_pong_feedback, 0.0, 1.0)
                _, effects.ping_pong_mix = imgui.slider_float("Mix", effects.ping_pong_mix, 0.0, 1.0)

            if current_effect == 1:  # Wah-wah
                my_text_separator("Wah-wah Settings")

                old_min_freq = effects.wah_wah_min_freq
                old_max_freq = effects.wah_wah_max_freq

                _, effects.wah_wah_damp = imgui.slider_float("Damping", effects.wah_wah_damp, 0.0, 1.0)
                _, effects.wah_wah_min_freq = imgui.slider_float("Min frequency (Hz)", effects.wah_wah_min_freq, 0.0, 10000.0)
                _, effects.wah_wah_max_freq = imgui.slider_float("Max frequency (Hz)", effects.wah_wah_max_freq, 0.0, 10000.0)
                _, effects.wah_wah_wah_freq = imgui.slider_float("Wah frequency (Hz)", effects.wah_wah_wah_freq, 0.0, 10000.0)

                if old_min_freq != effects.wah_wah_min_freq and effects.wah_wah_min_freq > effects.wah_wah_max_freq:
                    effects.wah_wah_max_freq = effects.wah_wah_min_freq + 1
                if old_max_freq != effects.wah_wah_max_freq and effects.wah_wah_max_freq < effects.wah_wah_min_freq:
                    effects.wah_wah_min_freq = effects.wah_wah_max_freq - 1

            if current_effect == 2:  # Flanger
                my_text_separator("Flanger Settings")

                _, effects.flanger_freq = imgui.slider_float("Frequency (Hz)", effects.flanger_freq, 0.0, 1.0)
                _, effects.flanger_depth = imgui.slider_float("Depth", effects.flanger_depth, 0.0, 0.5)
                _, effects.flanger_mix = imgui.slider_float("Mix", effects.flanger_mix, 0.0, 1.0)

            if current_effect == 3:  # Phaser
                my_text_separator("Phaser Settings")

                _, effects.phaser_max_delay = imgui.slider_int("Max delay (ms)", effects.phaser_max_delay, 0.0, 100)
                _, effects.phaser_freq = imgui.slider_float("Frequency (Hz)", effects.phaser_freq, 0.0, 10.0)
                _, effects.phaser_gain = imgui.slider_float("Gain", effects.phaser_gain, 0.0, 1.0)

            if current_effect == 4:  # Overdrive
                my_text_separator("Overdrive Settings")

                _, effects.overdrive_threshold = imgui.slider_float("Threshold", effects.overdrive_threshold, 0.0, 1.0)

            if current_effect == 5:  # Distortion
                my_text_separator("Distortion Settings")

                _, effects.distortion_gain = imgui.slider_int("Gain", effects.distortion_gain, 1, 100)

            if current_effect == 6:  # Reverb
                my_text_separator("Reverb Settings")

                for ir_name in effects.reverb_ir_names:
                    if imgui.radio_button(ir_name, effects.reverb_ir_names.index(ir_name) == effects.reverb_ir_index):
                        effects.reverb_ir_index = effects.reverb_ir_names.index(ir_name)

                _, effects.reverb_mix = imgui.slider_float("Mix", effects.reverb_mix, 0.0, 1.0)

            if current_effect == 7:  # Bit crusher
                my_text_separator("Bit Crusher Settings")

                _, effects.bit_crusher_bits = imgui.slider_int("Bits", effects.bit_crusher_bits, 1, 16)

            if current_effect == 8:  # 8D Audio
                my_text_separator("8D Audio Settings")

                _, effects.eight_d_spin_speed = imgui.slider_float("Spin speed", effects.eight_d_spin_speed, 0.0, 1.0)

            if current_effect == 9:  # Vocal doubler
                my_text_separator("Vocal Doubler Settings")

                _, effects.vocal_doubler_delay = imgui.slider_float("Delay (ms)", effects.vocal_doubler_delay, 0.1, 50.0)
                _, effects.vocal_doubler_detune_cents = imgui.slider_int("Detune cents", effects.vocal_doubler_detune_cents, 1, 200)

            if imgui.button("Apply effect"):
                if len(list(sounds.keys())) == 0 or current_sound > len(list(sounds.keys())):
                    print("No sound selected!")
                else:
                    apply_effect_button_callback()

            imgui.end()

        # Settings window
        if show_settings_window:
            imgui.set_next_window_size(400, 200, imgui.ONCE)
            imgui.set_next_window_position(int((WINDOW_WIDTH - 400) / 2), int((WINDOW_HEIGHT - 200) / 2), imgui.ONCE)

            _, show_settings_window = imgui.begin("Settings", True, imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_RESIZE)

            my_text_separator("Window Settings")
            if imgui.button("Set 1280x720"):
                WINDOW_WIDTH = 1280
                WINDOW_HEIGHT = 720
                glfw.set_window_size(window, WINDOW_WIDTH, WINDOW_HEIGHT)
                glfw.set_window_pos(window, int((mode.size.width - WINDOW_WIDTH) / 2),
                                    int((mode.size.height - WINDOW_HEIGHT) / 2))
            imgui.same_line()
            if imgui.button("Set 1600x900"):
                WINDOW_WIDTH = 1600
                WINDOW_HEIGHT = 900
                glfw.set_window_size(window, WINDOW_WIDTH, WINDOW_HEIGHT)
                glfw.set_window_pos(window, int((mode.size.width - WINDOW_WIDTH) / 2),
                                    int((mode.size.height - WINDOW_HEIGHT) / 2))
            imgui.same_line()
            if imgui.button("Set 1920x1080"):
                WINDOW_WIDTH = 1920
                WINDOW_HEIGHT = 1080
                glfw.set_window_size(window, WINDOW_WIDTH, WINDOW_HEIGHT)
                glfw.set_window_pos(window, int((mode.size.width - WINDOW_WIDTH) / 2),
                                    int((mode.size.height - WINDOW_HEIGHT) / 2))

            my_text_separator("Style Settings")
            _, current_style = imgui.combo("Style", current_style, ["Dark", "Light", "Classic"])

            if current_style == 0:
                imgui.style_colors_dark()
                background_color = (29. / 255, 29. / 255, 29. / 255)
            elif current_style == 1:
                imgui.style_colors_light()
                background_color = (240. / 255, 240. / 255, 240. / 255)
            elif current_style == 2:
                imgui.style_colors_classic()
                background_color = (38. / 255, 38. / 255, 38. / 255)

            imgui.end()

        # About window
        if show_about_window:
            imgui.set_next_window_size(300, 100, imgui.ALWAYS)
            imgui.set_next_window_position(int((WINDOW_WIDTH - 300) / 2), int((WINDOW_HEIGHT - 100) / 2), imgui.ALWAYS)

            _, show_about_window = imgui.begin(
                "About", True,
                imgui.WINDOW_ALWAYS_AUTO_RESIZE | imgui.WINDOW_NO_SAVED_SETTINGS | imgui.WINDOW_NO_NAV |
                imgui.WINDOW_NO_COLLAPSE
            )
            imgui.text("This application was made by:\nDominik Zappe")

            imgui.end()

        glClearColor(background_color[0], background_color[1], background_color[2], 1)
        glClear(GL_COLOR_BUFFER_BIT)

        imgui.render()
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)

    # Cleanup
    impl.shutdown()
    glfw.terminate()


if __name__ == "__main__":
    main()
