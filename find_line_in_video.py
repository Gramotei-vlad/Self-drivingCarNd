import matplotlib.image as mpimg
import numpy as np
from moviepy.editor import VideoFileClip
import cv2

def grayscale(img):
    """
    Трансформируем наше изображение в один цветовой канал,
    то есть проще говоря, в бесцветное.
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Делаем Canny transformation"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Сглаживание Гаусса. Удаляем шумы на изображении """
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Так как мы определяем линии на дороге, то
    нам нужна не нужна вся картинка, а только
    её определенная область. Поэтому обрезаем
    """
    mask = np.zeros_like(img)

    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255, 0, 0) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)

    masked_image = cv2.bitwise_and(img, mask)

    return masked_image


def draw_lines(img, lines, color, thickness=2):
    """
    Для отрисовки линий в изображении/видео
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    Возвращает изображение с hough lines.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    red = [255, 0, 0]
    draw_lines(line_img, lines, red)
    return line_img


def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    initial_img * α + img * β + γ
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)


def process_image(image):
    gray = grayscale(image)

    kernel_size = 7
    blur_gray = gaussian_blur(gray, kernel_size)

    low_threshold = 70
    high_threshold = 150
    edges = canny(blur_gray, low_threshold, high_threshold)

    imshape = image.shape
    vertices = np.array([[(0, imshape[0]), (520, 310), (600, 380), (imshape[1], imshape[0])]], dtype=np.int32)
    masked_image = region_of_interest(edges, vertices)

    rho = 1
    theta = np.pi / 180
    threshold = 1
    min_line_length = 4
    max_line_gap = 3
    line_image = hough_lines(masked_image, rho, theta, threshold, min_line_length, max_line_gap)

    result = weighted_img(line_image, image)

    return result

clip = VideoFileClip('test_images/solidWhiteRight.mp4')
white_clip = clip.fl_image(process_image)
white_clip.write_videofile('solidWhiteRight.mp4', audio=False)
