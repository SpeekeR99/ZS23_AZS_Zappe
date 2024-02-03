import numpy as np
import cv2
from OpenGL.GL import *
import glfw
import imgui
from imgui.integrations.glfw import GlfwRenderer
import wx

#  Window width
WINDOW_WIDTH = 1280
#  Window height
WINDOW_HEIGHT = 720

#  Whether to show the settings window
show_settings_window = False
#  Whether to show the about window
show_about_window = False
#  Whether to show the edge detection window
show_edge_detection_window = False
#  Whether to show the blur window
show_blur_window = False
#  Whether to show the threshold window
show_threshold_window = False
#  Whether to show the save as dialog window
show_save_as_dialog = False

#  Loaded images
imgs = {}
#  Currently selected image
current_img = 0

#  Edge detection methods
edge_detection_methods = ["Defined Direction Edge Detection", "Gradient Magnitude Direction Edge Detection",
                          "Mask Methods", "Laplacian Operator", "Line Detection", "Point Detection",
                          "Canny Edge Detection", "Canny Edge Detection (OpenCV)", "Marr-Hildreth Edge Detection"]
#  Currently selected edge detection method
current_edge_detection_method = 0
#  Kernel size for Gaussian blur
blur_kernel_size = 3
#  Whether to use Otsu threshold
otsu_threshold = False
#  Current threshold value
threshold_value = 127

#  Currently selected direction method
current_defined_direction_method = 0
#  Whether to use horizontal direction
defined_direction_horizontal = True
#  Whether to use vertical direction
defined_direction_vertical = False

#  Whether to use forward difference
forward_difference = True
#  Whether to use backward difference
backward_difference = False

#  Whether to use laplacian square or cross
laplacian_square = True

#  Current mask size
mask_size = 3
#  Default mask for size 2x2
default_mask_2 = np.array([[1, 0], [0, -1]])
#  Default mask for size 3x3
default_mask_3 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
#  Default mask for size 5x5
default_mask_5 = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, -24, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
#  Currently created mask
mask_methods_kernel = default_mask_3

#  Current point detection threshold
point_detection_threshold = 240

#  Current sigma for Gaussian blur in Canny edge detection
canny_sigma = 2
#  Current lower threshold for Canny edge detection
canny_lower_thresh = 20
#  Current upper threshold for Canny edge detection
canny_upper_thresh = 50

#  Current sigma for Gaussian blur in Marr-Hildreth edge detection
marr_hildreth_sigma = 2


def impl_glfw_init():
    """
    Initialize glfw and return the window
    :return: Window object
    """
    window_name = "Edge Detection Semestral Work"

    if not glfw.init():
        print("Could not initialize OpenGL context")
        exit(1)

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
    glfw.window_hint(glfw.RESIZABLE, GL_FALSE)

    window = glfw.create_window(int(WINDOW_WIDTH), int(WINDOW_HEIGHT), window_name, None, None)
    glfw.make_context_current(window)
    mode = glfw.get_video_mode(glfw.get_primary_monitor())
    glfw.set_window_pos(window, int((mode.size.width - WINDOW_WIDTH) / 2), int((mode.size.height - WINDOW_HEIGHT) / 2))

    if not window:
        glfw.terminate()
        print("Could not initialize Window")
        exit(1)

    return window


def texture_image(img):
    """
    Create texture from image
    Texture is needed in order to show a picture that is resizable,
    but remain the original aspect ratio and picture for the backend processing
    :param img: Image to create texture from
    :return: Texture ID
    """
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.shape[1], img.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, img)
    glBindTexture(GL_TEXTURE_2D, 0)
    return texture


def show_image(name):
    """
    Adds an image to the global dictionary of images
    Creates an ImGui window for the image texture
    :param name: Name of the image
    :return: Boolean whether to close the window or not
    """
    global imgs
    imgui.set_next_window_size(imgs[name]["render_img"].shape[1] + 15, imgs[name]["render_img"].shape[0] + 35,
                               imgui.ONCE)
    _, close_bool = imgui.begin(name, True, imgui.WINDOW_NO_SAVED_SETTINGS | imgui.WINDOW_NO_COLLAPSE)
    window_size = imgui.get_window_size()
    dx = window_size[0] - 15 - imgs[name]["original_size"][0]
    dy = window_size[1] - 35 - imgs[name]["original_size"][1]
    scale = min((imgs[name]["original_size"][0] + dx) / (imgs[name]["original_size"][0]),
                (imgs[name]["original_size"][1] + dy) / (imgs[name]["original_size"][1]))
    scale = max(scale, 0.01)
    if dx or dy:
        imgs[name]["render_img"] = cv2.resize(imgs[name]["render_img"], (
            int(imgs[name]["original_size"][0] * scale), int(imgs[name]["original_size"][1] * scale)))
    imgui.image(imgs[name]["texture"], imgs[name]["render_img"].shape[1], imgs[name]["render_img"].shape[0])
    imgui.end()
    return close_bool


def create_render_img_and_texture(img):
    """
    Creates a render image and texture from the original image
    :param img: Image to create render image and texture from
    :return: Render image and texture
    """
    render_img = np.copy(img)
    texture = texture_image(render_img)
    dx = WINDOW_WIDTH * 0.4 - render_img.shape[0]
    dy = WINDOW_HEIGHT * 0.4 - render_img.shape[1]
    if abs(dx) < abs(dy):
        scale = (render_img.shape[0] + dx) / (render_img.shape[0])
    else:
        scale = (render_img.shape[1] + dy) / (render_img.shape[1])
    render_img = cv2.resize(render_img, (int(render_img.shape[1] * scale), int(render_img.shape[0] * scale)))
    return render_img, texture


def avoid_name_duplicates(filepath):
    """
    Checks if the image name already exists in the global dictionary of images
    If it does, it adds a number to the end of the name
    :param filepath: Filepath of the image
    :return: Unique name of the image
    """
    name = filepath.split("/")[-1].split(".")[0]
    extension = filepath.split("/")[-1].split(".")[-1]
    copy = 1
    while name + "." + extension in imgs:
        if copy == 1:
            name = name + " (1)"
        else:
            name = name[:-3]
            name = name + "(" + str(copy) + ")"
        copy += 1
    return name + "." + extension


def load_image(filepath):
    """
    Loads an image from the given filepath
    :param filepath: Filepath of the image
    :return: None
    """
    global imgs
    filepath = filepath.replace("\\", "/")
    img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    if img is None:
        print("Error loading image: " + filepath + "!")
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    render_img, texture = create_render_img_and_texture(img)
    name = avoid_name_duplicates(filepath)
    imgs[name] = {"img": img, "render_img": render_img, "texture": texture, "show": True,
                  "original_size": (img.shape[1], img.shape[0])}


def my_text_separator(text):
    """
    Creates a separator with text in the middle
    :param text: Text to display
    :return: None
    """
    imgui.separator()
    imgui.text(text)
    imgui.separator()


def defined_direction_edge_detection(img):
    """
    Performs edge detection on the given image using the defined direction method
    :param img: Image to perform edge detection on
    :return: Image with edges
    """
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img.astype(np.float32)
    horizontal_kernel_1 = np.ones((3, 3))
    horizontal_kernel_2 = np.ones((3, 3))
    vertical_kernel_1 = np.ones((3, 3))
    vertical_kernel_2 = np.ones((3, 3))

    if current_defined_direction_method == 0:  # Sobel
        horizontal_kernel_1 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        horizontal_kernel_2 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        vertical_kernel_1 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        vertical_kernel_2 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    elif current_defined_direction_method == 1:  # Prewitt
        horizontal_kernel_1 = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        horizontal_kernel_2 = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        vertical_kernel_1 = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        vertical_kernel_2 = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    elif current_defined_direction_method == 2:  # Roberts
        horizontal_kernel_1 = np.array([[1, 0], [0, -1]])
        horizontal_kernel_2 = np.array([[0, 1], [-1, 0]])
        vertical_kernel_1 = np.array([[0, 1], [-1, 0]])
        vertical_kernel_2 = np.array([[1, 0], [0, -1]])

    horizontal_1 = cv2.filter2D(img, -1, horizontal_kernel_1)
    horizontal_2 = cv2.filter2D(img, -1, horizontal_kernel_2)
    vertical_1 = cv2.filter2D(img, -1, vertical_kernel_1)
    vertical_2 = cv2.filter2D(img, -1, vertical_kernel_2)

    if defined_direction_horizontal:
        res = np.sqrt(np.power(horizontal_1, 2) + np.power(horizontal_2, 2))
    elif defined_direction_vertical:
        res = np.sqrt(np.power(vertical_1, 2) + np.power(vertical_2, 2))
    else:
        res = np.sqrt(
            np.power(horizontal_1, 2) + np.power(horizontal_2, 2) + np.power(vertical_1, 2) + np.power(vertical_2, 2))

    res = res / np.max(res) * 255
    res = res.astype(np.uint8)
    return res


def gradient_magnitude_direction_edge_detection(img):
    """
    Performs edge detection on the given image using the gradient magnitude and direction method
    :param img: Image to perform edge detection on
    :return: Image with edges
    """
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = np.float32(img)

    forward_difference_x = np.zeros(img.shape, dtype=np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1] - 1):
            forward_difference_x[i, j] = abs(img[i, j + 1] - img[i, j])
    forward_difference_y = np.zeros(img.shape, dtype=np.float32)
    for i in range(img.shape[0] - 1):
        for j in range(img.shape[1]):
            forward_difference_y[i, j] = abs(img[i + 1, j] - img[i, j])
    backward_difference_x = np.zeros(img.shape, dtype=np.float32)
    for i in range(img.shape[0]):
        for j in range(1, img.shape[1]):
            backward_difference_x[i, j] = abs(img[i, j] - img[i, j - 1])
    backward_difference_y = np.zeros(img.shape, dtype=np.float32)
    for i in range(1, img.shape[0]):
        for j in range(img.shape[1]):
            backward_difference_y[i, j] = abs(img[i, j] - img[i - 1, j])
    central_difference_x = backward_difference_x + forward_difference_x / 2
    central_difference_y = backward_difference_y + forward_difference_y / 2

    forward_difference_x = forward_difference_x.astype(np.uint8)
    forward_difference_y = forward_difference_y.astype(np.uint8)
    backward_difference_x = backward_difference_x.astype(np.uint8)
    backward_difference_y = backward_difference_y.astype(np.uint8)
    central_difference_x = central_difference_x.astype(np.uint8)
    central_difference_y = central_difference_y.astype(np.uint8)

    if forward_difference:
        if defined_direction_horizontal:
            return forward_difference_x
        elif defined_direction_vertical:
            return forward_difference_y
        else:
            return forward_difference_x + forward_difference_y
    elif backward_difference:
        if defined_direction_horizontal:
            return backward_difference_x
        elif defined_direction_vertical:
            return backward_difference_y
        else:
            return backward_difference_x + backward_difference_y
    else:
        if defined_direction_horizontal:
            return central_difference_x
        elif defined_direction_vertical:
            return central_difference_y
        else:
            return central_difference_x + central_difference_y


def mask_methods_edge_detection(img):
    """
    Performs edge detection on the given image using the mask methods
    :param img: Image to perform edge detection on
    :return: Image with edges
    """
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    res = cv2.filter2D(img, -1, mask_methods_kernel)
    res = res / np.max(res) * 255
    res = res.astype(np.uint8)
    return res


def laplacian_operator_edge_detection(img):
    """
    Performs edge detection on the given image using the Laplacian operator
    :param img: Image to perform edge detection on
    :return: Image with edges
    """
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = np.ones((3, 3))

    if laplacian_square:
        kernel[1, 1] = -8
    else:
        kernel[0, 0] = 0
        kernel[0, 2] = 0
        kernel[2, 0] = 0
        kernel[2, 2] = 0
        kernel[1, 1] = -4
    kernel = -1 * kernel

    res = cv2.filter2D(img, -1, kernel)
    return res


def line_detection_edge_detection(img):
    """
    Performs line detection on the given image using the Hough transform
    :param img: Image to perform line detection on
    :return: Image with red lines on it
    """
    res = np.copy(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    lines = cv2.HoughLinesP(img, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(res, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA)

    return res


def point_detection_edge_detection(img):
    """
    Performs point detection on the given image
    :param img: Image to perform point detection on
    :return: Image with red points on it
    """
    res = np.copy(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = np.array([[-1 / 8, -1 / 8, -1 / 8], [-1 / 8, 1, -1 / 8], [-1 / 8, -1 / 8, -1 / 8]])

    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            if np.sum(img[i - 1:i + 2, j - 1:j + 2] * mask) > point_detection_threshold:
                res[i, j] = [0, 0, 255]

    return res


def canny_edge_detection(img):
    """
    Performs edge detection on the given image using the Canny edge detector
    :param img: Image to perform edge detection on
    :return: Image with edges
    """
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    size = int(2 * (np.ceil(3 * canny_sigma)) + 1)
    x, y = np.meshgrid(np.arange(-size / 2 + 1, size / 2 + 1),
                       np.arange(-size / 2 + 1, size / 2 + 1))
    normal = 1 / (2.0 * np.pi * canny_sigma ** 2)
    kernel = np.exp(-(x ** 2 + y ** 2) / (2.0 * canny_sigma ** 2)) / normal
    kern_size, gauss = kernel.shape[0], np.zeros_like(img, dtype=float)

    for i in range(img.shape[0] - (kern_size - 1)):
        for j in range(img.shape[1] - (kern_size - 1)):
            window = img[i:i + kern_size, j:j + kern_size] * kernel
            gauss[i, j] = np.sum(window)

    kernel, kern_size = np.array(
        [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]), 3
    gx, gy = np.zeros_like(
        gauss, dtype=float), np.zeros_like(gauss, dtype=float)

    for i in range(gauss.shape[0] - (kern_size - 1)):
        for j in range(gauss.shape[1] - (kern_size - 1)):
            window = gauss[i:i + kern_size, j:j + kern_size]
            gx[i, j], gy[i, j] = np.sum(
                window * kernel.T), np.sum(window * kernel)

    gradient = np.sqrt(gx ** 2 + gy ** 2)
    theta = np.arctan2(gy, gx)
    theta += np.pi * np.int32(theta < 0)
    non_max_suppression = np.copy(gradient)

    for j in range(1, gradient.shape[0] - 1):
        for i in range(1, gradient.shape[1] - 1):
            if (0 <= theta[j, i] < 22.5 / 180 * np.pi) or (157.5 / 180 * np.pi <= theta[j, i] < np.pi):
                if gradient[j, i] < gradient[j, i - 1] or gradient[j, i] < gradient[j, i + 1]:
                    non_max_suppression[j, i] = 0
            elif 22.5 / 180 * np.pi <= theta[j, i] < 67.5 / 180 * np.pi:
                if gradient[j, i] < gradient[j - 1, i - 1] or gradient[j, i] < gradient[j + 1, i + 1]:
                    non_max_suppression[j, i] = 0
            elif 67.5 / 180 * np.pi <= theta[j, i] < 112.5 / 180 * np.pi:
                if gradient[j, i] < gradient[j - 1, i] or gradient[j, i] < gradient[j + 1, i]:
                    non_max_suppression[j, i] = 0
            elif 112.5 / 180 * np.pi <= theta[j, i] < 157.5 / 180 * np.pi:
                if gradient[j, i] < gradient[j + 1, i - 1] or gradient[j, i] < gradient[j - 1, i + 1]:
                    non_max_suppression[j, i] = 0

    non_max_suppression = non_max_suppression / np.max(non_max_suppression) * 255
    weak, strong = np.copy(non_max_suppression), np.copy(non_max_suppression)
    weak[weak < canny_lower_thresh] = 0
    weak[weak >= canny_upper_thresh] = 0
    weak[weak != 0] = 255
    strong[strong < canny_upper_thresh] = 0
    strong[strong >= canny_upper_thresh] = 255

    res = weak + strong
    res = res.astype(np.uint8)

    return res


def canny_edge_detection_opencv(img):
    """
    Performs edge detection on the given image using the Canny edge detector (OpenCV implementation)
    :param img: Image to perform edge detection on
    :return: Image with edges
    """
    return cv2.Canny(img, canny_lower_thresh, canny_upper_thresh)


def marr_hildreth_edge_detection(img):
    """
    Performs edge detection on the given image using the Marr-Hildreth edge detector
    :param img: Image to perform edge detection on
    :return: Image with edges
    """
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    size = int(2 * (np.ceil(3 * marr_hildreth_sigma)) + 1)
    x, y = np.meshgrid(np.arange(-size / 2 + 1, size / 2 + 1),
                       np.arange(-size / 2 + 1, size / 2 + 1))
    normal = 1 / (2.0 * np.pi * marr_hildreth_sigma ** 2)
    kernel = ((x ** 2 + y ** 2 - (2.0 * marr_hildreth_sigma ** 2)) / marr_hildreth_sigma ** 4) * np.exp(
        -(x ** 2 + y ** 2) / (2.0 * marr_hildreth_sigma ** 2)) / normal
    kern_size = kernel.shape[0]
    log = np.zeros_like(img, dtype=float)

    for i in range(img.shape[0] - (kern_size - 1)):
        for j in range(img.shape[1] - (kern_size - 1)):
            window = img[i:i + kern_size, j:j + kern_size] * kernel
            log[i, j] = np.sum(window)

    log = log.astype(np.int64, copy=False)
    zero_crossing = np.zeros_like(log)

    for i in range(log.shape[0] - (kern_size - 1)):
        for j in range(log.shape[1] - (kern_size - 1)):
            if log[i][j] == 0:
                if (log[i][j - 1] < 0 and log[i][j + 1] > 0) or (log[i][j - 1] < 0 and log[i][j + 1] < 0) or (
                        log[i - 1][j] < 0 and log[i + 1][j] > 0) or (log[i - 1][j] > 0 and log[i + 1][j] < 0):
                    zero_crossing[i][j] = 255
            if log[i][j] < 0:
                if (log[i][j - 1] > 0) or (log[i][j + 1] > 0) or (log[i - 1][j] > 0) or (log[i + 1][j] > 0):
                    zero_crossing[i][j] = 255

    zero_crossing = zero_crossing.astype(np.uint8)
    return zero_crossing


def generate_button_callback():
    """
    Callback for the generate button
    Creates new image with the selected options applied to it
    :return: None
    """
    global imgs
    edge_detection = [defined_direction_edge_detection, gradient_magnitude_direction_edge_detection,
                      mask_methods_edge_detection, laplacian_operator_edge_detection, line_detection_edge_detection,
                      point_detection_edge_detection, canny_edge_detection, canny_edge_detection_opencv, marr_hildreth_edge_detection]
    img = imgs[list(imgs.keys())[current_img]]["img"]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    res = edge_detection[current_edge_detection_method](img)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    render_res, texture = create_render_img_and_texture(res)
    name = avoid_name_duplicates(list(imgs.keys())[current_img].split(".")[0] + " (" + edge_detection_methods[
        current_edge_detection_method] + ")." + list(imgs.keys())[current_img].split(".")[-1])
    imgs[name] = {"img": res, "render_img": render_res, "texture": texture,
                  "show": True, "original_size": (res.shape[1], res.shape[0])}


def blur_button_callback():
    """
    Callback for the blur button
    Blurs the current image with the selected blur kernel size
    :return: None
    """
    global imgs, threshold_value
    img = imgs[list(imgs.keys())[current_img]]["img"]
    res = cv2.GaussianBlur(img, (blur_kernel_size, blur_kernel_size), 0)
    render_res, texture = create_render_img_and_texture(res)
    name = avoid_name_duplicates(
        list(imgs.keys())[current_img].split(".")[0] + " (Blurred)." + list(imgs.keys())[current_img].split(".")[-1])
    imgs[name] = {"img": res, "render_img": render_res, "texture": texture,
                  "show": True, "original_size": (res.shape[1], res.shape[0])}


def threshold_button_callback():
    """
    Callback for the threshold button
    Thresholds the current image with the selected threshold value
    :return: None
    """
    global imgs, threshold_value
    img = imgs[list(imgs.keys())[current_img]]["img"]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if otsu_threshold:
        threshold_value, res = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        res = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)[1]
    res = cv2.cvtColor(res, cv2.COLOR_GRAY2RGB)
    render_res, texture = create_render_img_and_texture(res)
    name = avoid_name_duplicates(
        list(imgs.keys())[current_img].split(".")[0] + " (Threshold)." + list(imgs.keys())[current_img].split(".")[-1])
    imgs[name] = {"img": res, "render_img": render_res, "texture": texture,
                  "show": True, "original_size": (res.shape[1], res.shape[0])}


def main():
    """
    Main function
    """
    global WINDOW_WIDTH, WINDOW_HEIGHT, show_settings_window, show_about_window, show_edge_detection_window, \
        show_blur_window, blur_kernel_size, show_threshold_window, show_save_as_dialog, imgs, current_img, \
        current_edge_detection_method, otsu_threshold, threshold_value, laplacian_square, \
        current_defined_direction_method, defined_direction_horizontal, defined_direction_vertical, mask_size, \
        mask_methods_kernel, forward_difference, backward_difference, point_detection_threshold, canny_sigma, \
        canny_lower_thresh, canny_upper_thresh, marr_hildreth_sigma

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
                        imgs = {}
                        show_edge_detection_window = False
                        show_about_window = False
                        show_settings_window = False

                clicked_load, _ = imgui.menu_item("Load Image...", None, False, True)
                if clicked_load:
                    filepath = wx.FileSelector(
                        "Load Image",
                        wildcard="Image Files (*.png;*.jpg;*.jpeg;*.bmp)|*.png;*.jpg;*.jpeg;*.bmp"
                    )
                    if filepath:
                        load_image(filepath)

                clicked_save, _ = imgui.menu_item("Save Image as...", None, False, True)
                if clicked_save:
                    show_save_as_dialog = True

                imgui.separator()

                clicked_exit, _ = imgui.menu_item("Exit", 'Alt+F4', False, True)
                if clicked_exit:
                    glfw.set_window_should_close(window, True)

                imgui.end_menu()
            if imgui.begin_menu("Edit"):
                clicked_edge_detect, _ = imgui.menu_item("Edge Detection...", None, False, True)
                if clicked_edge_detect:
                    show_edge_detection_window = True

                imgui.separator()

                clicked_blur, _ = imgui.menu_item("Blur...", None, False, True)
                if clicked_blur:
                    show_blur_window = True

                clicked_threshold, _ = imgui.menu_item("Threshold...", None, False, True)
                if clicked_threshold:
                    show_threshold_window = True

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
            glDeleteTextures(1, [imgs[to_be_deleted]["texture"]])
            imgs.pop(to_be_deleted)
            to_be_deleted = None

        for name in imgs:
            if imgs[name]["show"]:
                imgs[name]["show"] = show_image(name)
                if not imgs[name]["show"]:
                    sure = wx.MessageDialog(
                        None,
                        "Are you sure you want to delete " + name + "? All unsaved changes will be lost!",
                        "Delete Image", wx.YES_NO | wx.ICON_QUESTION
                    ).ShowModal()
                    if sure == wx.ID_YES:
                        to_be_deleted = name
                    else:
                        imgs[name]["show"] = True

        if show_save_as_dialog:
            imgui.set_next_window_size(500, 100, imgui.ONCE)
            imgui.set_next_window_position((WINDOW_WIDTH - 500) / 2, (WINDOW_HEIGHT - 100) / 2, imgui.ONCE)

            _, show_save_as_dialog = imgui.begin("Save Image as...", True, imgui.WINDOW_NO_COLLAPSE)

            imgui.text("Image Selection:")
            _, current_img = imgui.combo("Image", current_img, list(imgs.keys()))

            if imgui.button("Save as..."):
                if len(list(imgs.keys())) == 0 or current_img > len(list(imgs.keys())):
                    print("No image selected!")
                else:
                    filepath = wx.FileSelector(
                        "Save Image as...", default_filename=list(imgs.keys())[current_img],
                        wildcard="Image Files (*.png;*.jpg;*.jpeg;*.bmp)|*.png;*.jpg;*.jpeg;*.bmp",
                        flags=wx.FD_SAVE
                    )
                    if filepath:
                        img = imgs[list(imgs.keys())[current_img]]["img"]
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        cv2.imwrite(filepath, img)

            imgui.end()

        if show_edge_detection_window:
            imgui.set_next_window_size(500, 500, imgui.ONCE)
            imgui.set_next_window_position((WINDOW_WIDTH - 500) / 2, (WINDOW_HEIGHT - 500) / 2, imgui.ONCE)

            _, show_edge_detection_window = imgui.begin("Edge Detection", True, imgui.WINDOW_NO_COLLAPSE)

            my_text_separator("Image Selection")
            _, current_img = imgui.combo("Image", current_img, list(imgs.keys()))

            my_text_separator("Edge Detection Method")
            _, current_edge_detection_method = imgui.combo("Edge Detection Method", current_edge_detection_method,
                                                           edge_detection_methods)

            if current_edge_detection_method == 0:  # Defined Direction
                _, current_defined_direction_method = imgui.combo("Method", current_defined_direction_method,
                                                                  ["Sobel", "Prewitt", "Roberts"])
                imgui.text("Direction:")
                if imgui.radio_button("Horizontal", defined_direction_horizontal):
                    defined_direction_horizontal = True
                    defined_direction_vertical = False
                if imgui.radio_button("Vertical", defined_direction_vertical):
                    defined_direction_horizontal = False
                    defined_direction_vertical = True
                if imgui.radio_button("Both", not defined_direction_horizontal and not defined_direction_vertical):
                    defined_direction_horizontal = False
                    defined_direction_vertical = False

            elif current_edge_detection_method == 1:  # Gradient Magnitude
                imgui.text("Difference Method:")
                if imgui.radio_button("Forward", forward_difference):
                    forward_difference = True
                    backward_difference = False
                if imgui.radio_button("Backward", backward_difference):
                    forward_difference = False
                    backward_difference = True
                if imgui.radio_button("Central", not forward_difference and not backward_difference):
                    forward_difference = False
                    backward_difference = False
                imgui.text("Direction:")
                if imgui.radio_button("Horizontal", defined_direction_horizontal):
                    defined_direction_horizontal = True
                    defined_direction_vertical = False
                if imgui.radio_button("Vertical", defined_direction_vertical):
                    defined_direction_horizontal = False
                    defined_direction_vertical = True
                if imgui.radio_button("Both", not defined_direction_horizontal and not defined_direction_vertical):
                    defined_direction_horizontal = False
                    defined_direction_vertical = False

            elif current_edge_detection_method == 2:  # Mask Methods
                mask_size_changed, mask_size = imgui.slider_int("Mask Size", mask_size, 2, 5)
                if mask_size_changed:
                    if mask_size % 2 == 0 and mask_size != 2:
                        mask_size += 1
                    if mask_size == 2:
                        mask_methods_kernel = default_mask_2
                    elif mask_size == 3:
                        mask_methods_kernel = default_mask_3
                    elif mask_size == 5:
                        mask_methods_kernel = default_mask_5
                imgui.text("Mask:")
                imgui.push_style_var(imgui.STYLE_ITEM_SPACING, (5, 5))
                imgui.push_item_width(30)
                for i in range(0, mask_size):
                    for j in range(0, mask_size):
                        _, mask_methods_kernel[i][j] = imgui.input_int("##" + str(i) + str(j),
                                                                       mask_methods_kernel[i][j], 0, 0)
                        if j != mask_size - 1:
                            imgui.same_line()
                imgui.pop_item_width()
                imgui.pop_style_var()

            elif current_edge_detection_method == 3:  # Laplacian Operator
                imgui.text("Laplacian Kernel Type:")
                if imgui.radio_button("Cross", not laplacian_square):
                    laplacian_square = False
                if imgui.radio_button("Square", laplacian_square):
                    laplacian_square = True

            elif current_edge_detection_method == 4:  # Line Detection
                pass

            elif current_edge_detection_method == 5:  # Point Detection
                _, point_detection_threshold = imgui.slider_int("Threshold", point_detection_threshold, 0, 255)

            elif current_edge_detection_method == 6:  # Canny Edge Detection
                _, canny_sigma = imgui.slider_float("Sigma", canny_sigma, 0.1, 10.0, "%.1f")
                imgui.text("Canny Thresholds:")
                old_lower = canny_lower_thresh
                old_upper = canny_upper_thresh
                _, canny_lower_thresh = imgui.slider_int("Lower Threshold", canny_lower_thresh, 0, 254)
                _, canny_upper_thresh = imgui.slider_int("Upper Threshold", canny_upper_thresh, 1, 255)

                if old_lower != canny_lower_thresh and canny_lower_thresh >= canny_upper_thresh:
                    canny_upper_thresh = canny_lower_thresh + 1
                if old_upper != canny_upper_thresh and canny_upper_thresh <= canny_lower_thresh:
                    canny_lower_thresh = canny_upper_thresh - 1

            elif current_edge_detection_method == 7:  # Canny Edge Detection (OpenCV)
                imgui.text("Canny Thresholds:")
                old_lower = canny_lower_thresh
                old_upper = canny_upper_thresh
                _, canny_lower_thresh = imgui.slider_int("Lower Threshold", canny_lower_thresh, 0, 254)
                _, canny_upper_thresh = imgui.slider_int("Upper Threshold", canny_upper_thresh, 1, 255)

                if old_lower != canny_lower_thresh and canny_lower_thresh >= canny_upper_thresh:
                    canny_upper_thresh = canny_lower_thresh + 1
                if old_upper != canny_upper_thresh and canny_upper_thresh <= canny_lower_thresh:
                    canny_lower_thresh = canny_upper_thresh - 1

            elif current_edge_detection_method == 8:  # Marr-Hildreth Edge Detection
                _, marr_hildreth_sigma = imgui.slider_float("Sigma", marr_hildreth_sigma, 0.1, 10.0, "%.1f")

            if imgui.button("Generate"):
                if len(list(imgs.keys())) == 0 or current_img > len(list(imgs.keys())):
                    print("No image selected!")
                else:
                    generate_button_callback()

            imgui.end()

        if show_blur_window:
            imgui.set_next_window_size(500, 500, imgui.ONCE)
            imgui.set_next_window_position((WINDOW_WIDTH - 500) / 2, (WINDOW_HEIGHT - 500) / 2, imgui.ONCE)

            _, show_blur_window = imgui.begin("Blurring", True, imgui.WINDOW_NO_COLLAPSE)

            my_text_separator("Image Selection")
            _, current_img = imgui.combo("Image", current_img, list(imgs.keys()))

            my_text_separator("Blurring Settings")
            changed, blur_kernel_size = imgui.slider_int("Kernel Size", blur_kernel_size, 3, 51)
            if changed and blur_kernel_size % 2 == 0:
                blur_kernel_size += 1
            if imgui.button("Blur"):
                if len(list(imgs.keys())) == 0 or current_img > len(list(imgs.keys())):
                    print("No image selected!")
                else:
                    blur_button_callback()

            imgui.end()

        if show_threshold_window:
            imgui.set_next_window_size(500, 500, imgui.ONCE)
            imgui.set_next_window_position((WINDOW_WIDTH - 500) / 2, (WINDOW_HEIGHT - 500) / 2, imgui.ONCE)

            _, show_threshold_window = imgui.begin("Thresholding", True, imgui.WINDOW_NO_COLLAPSE)

            my_text_separator("Image Selection")
            _, current_img = imgui.combo("Image", current_img, list(imgs.keys()))

            my_text_separator("Thresholding Settings")
            _, otsu_threshold = imgui.checkbox("Otsu Threshold", otsu_threshold)
            if not otsu_threshold:
                _, threshold_value = imgui.slider_int("Threshold Value", threshold_value, 0, 255)

            if imgui.button("Threshold"):
                if len(list(imgs.keys())) == 0 or current_img > len(list(imgs.keys())):
                    print("No image selected!")
                else:
                    threshold_button_callback()

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
