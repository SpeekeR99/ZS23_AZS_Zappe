from gui import *
from effects import *
from utils import *
from imgui.integrations.glfw import GlfwRenderer
import wx


def main():
    """
    Main function
    """
    global WINDOW_WIDTH, WINDOW_HEIGHT, show_settings_window, show_about_window, show_save_as_dialog, sounds, \
        current_sound, show_wah_wah_window, wah_wah_damp, wah_wah_min_freq, wah_wah_max_freq, wah_wah_wah_freq

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

    while not glfw.window_should_close(window):
        glfw.poll_events()
        impl.process_inputs()

        imgui.new_frame()

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
                imgui.menu_item("Filtering effects", None, False, False)
                clicked_wah_wah, _ = imgui.menu_item("Wah-wah", None, False, True)
                if clicked_wah_wah:
                    show_wah_wah_window = True
                imgui.separator()

                imgui.menu_item("Modulation effects", None, False, False)
                clicked_flanger, _ = imgui.menu_item("Flanger", None, False, True)
                if clicked_flanger:
                    show_flanger_window = True
                clicked_phaser, _ = imgui.menu_item("Phaser", None, False, True)
                if clicked_phaser:
                    show_phaser_window = True
                imgui.separator()

                # imgui.menu_item("Frequency effects", None, False, False)
                # clicked_vibrato, _ = imgui.menu_item("Vibrato", None, False, True)
                # if clicked_vibrato:
                #     show_vibrato_window = True
                # imgui.separator()

                imgui.menu_item("Saturation effects", None, False, False)
                clicked_overdrive, _ = imgui.menu_item("Overdrive", None, False, True)
                if clicked_overdrive:
                    show_overdrive_window = True
                clicked_distortion, _ = imgui.menu_item("Distortion", None, False, True)
                if clicked_distortion:
                    show_distortion_window = True
                imgui.separator()

                imgui.menu_item("Time effects", None, False, False)
                clicked_reverb, _ = imgui.menu_item("Reverb", None, False, True)
                if clicked_reverb:
                    show_reverb_window = True
                imgui.separator()

                imgui.menu_item("Unclassified effects", None, False, False)
                clicked_bit_crusher, _ = imgui.menu_item("Bit crusher", None, False, True)
                if clicked_bit_crusher:
                    show_bit_crusher_window = True
                clicked_spin_around, _ = imgui.menu_item("Spin around (8D audio)", None, False, True)
                if clicked_spin_around:
                    show_spin_around_window = True

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

        if show_wah_wah_window:
            imgui.set_next_window_size(400, 300, imgui.ONCE)
            imgui.set_next_window_position(int((WINDOW_WIDTH - 400) / 2), int((WINDOW_HEIGHT - 200) / 2), imgui.ONCE)

            _, show_wah_wah_window = imgui.begin("Wah-wah", True, imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_RESIZE)

            my_text_separator("Sound selection")
            _, current_sound = imgui.combo("Smage", current_sound, list(sounds.keys()))

            my_text_separator("Wah-wah Settings")
            _, wah_wah_damp = imgui.slider_float("Damping", wah_wah_damp, 0.0, 1.0)
            freq_changed, wah_wah_min_freq = imgui.slider_float("Min frequency", wah_wah_min_freq, 0.0, 10000.0)
            freq_changed, wah_wah_max_freq = imgui.slider_float("Max frequency", wah_wah_max_freq, 0.0, 10000.0)
            _ , wah_wah_wah_freq = imgui.slider_float("Wah frequency", wah_wah_wah_freq, 0.0, 10000.0)

            if freq_changed and wah_wah_min_freq > wah_wah_max_freq:
                wah_wah_max_freq = wah_wah_min_freq + 1
            elif freq_changed and wah_wah_max_freq < wah_wah_min_freq:
                wah_wah_min_freq = wah_wah_max_freq - 1

            if imgui.button("Apply effect"):
                if len(list(sounds.keys())) == 0 or current_sound > len(list(sounds.keys())):
                    print("No sound selected!")
                else:
                    name = list(sounds.keys())[current_sound]
                    sound = sounds[name]
                    audio = sound["data"]
                    audio_l, audio_r = extract_channels(audio)
                    sample_rate = sound["sample_rate"]

                    out_audio_l, out_sample_rate = apply_wah_wah(audio_l, sample_rate, damp=wah_wah_damp, min_freq=wah_wah_min_freq, max_freq=wah_wah_max_freq, wah_freq=wah_wah_wah_freq)
                    out_audio_r, out_sample_rate = apply_wah_wah(audio_r, sample_rate, damp=wah_wah_damp, min_freq=wah_wah_min_freq, max_freq=wah_wah_max_freq, wah_freq=wah_wah_wah_freq)
                    out_audio = to_stereo(out_audio_l, out_audio_r)

                    new_name = avoid_name_duplicates(name.split(".")[0] + " (Wah-Wah)." + name.split(".")[-1])
                    sounds[new_name] = {"data": out_audio, "sample_rate": out_sample_rate, "show": True}

            imgui.end()

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

    impl.shutdown()
    glfw.terminate()


if __name__ == "__main__":
    main()
