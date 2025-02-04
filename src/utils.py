import cv2

def augment_image(image, blur_kernel_size=5, angle=18):
    """
    Аугментирует изображение, применяя размытие и поворот.

    :param image: Входное изображение в формате np.array.
    :param blur_kernel_size: Размер ядра для размытия (по умолчанию 5).
    :param angle: Угол поворота изображения (по умолчанию 45 градусов).
    :return: Аугментированное изображение в формате np.array.
    """
    # Применяем размытие
    image = cv2.imread(image)
    blurred_image = cv2.GaussianBlur(image, (blur_kernel_size, blur_kernel_size), 0)

    # Получаем размеры изображения
    (h, w) = blurred_image.shape[:2]

    # Вычисляем центр изображения
    center = (w // 2, h // 2)

    # Получаем матрицу поворота
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Применяем поворот
    rotated_image = cv2.warpAffine(blurred_image, M, (w, h))

    return rotated_image