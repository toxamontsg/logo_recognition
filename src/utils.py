from PIL import Image, ImageFilter
import random

def augment_image(image, blur_kernel_size_range=(3, 7), angle_range=(-15, 15)):
    # Генерация случайного угла поворота в заданном диапазоне
    random_angle = random.uniform(angle_range[0], angle_range[1])

    # Поворот изображения
    rotated_image = image.rotate(random_angle, expand=True)

    # Генерация случайного размера ядра размытия в заданном диапазоне
    random_blur_kernel_size = random.randint(blur_kernel_size_range[0], blur_kernel_size_range[1])

    # Применение размытия
    blurred_image = rotated_image.filter(ImageFilter.GaussianBlur(random_blur_kernel_size))

    return blurred_image