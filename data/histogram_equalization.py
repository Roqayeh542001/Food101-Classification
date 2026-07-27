import cv2
import numpy as np
from PIL import Image

class HistogramEqualization:
    """
    Histogram Equalization on the V channel of HSV image.
    Based on FoodNet paper.
    """

    def __call__(self, image: Image.Image):
        # PIL -> Numpy
        image = np.array(image)

        # RGB -> HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Equalization only Value channel
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])

        # HSV -> RGB
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        return Image.fromarray(rgb)