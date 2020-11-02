import numpy
from freefield import DIR
import cv2
import os


class VirtualCam:
    def __init__(self):
        self.ncams = 1
        self.imsize = self.acquire_images(n=1).shape[0:2]

    def acquire_images(self, n=1):

        image = numpy.random.choice(os.listdir(DIR/"test"/"images"))
        image = cv2.imread(image)
        if hasattr(self, "imagesize"):
            image_data = numpy.zeros((self.imsize)+(n, self.ncams),
                                     dtype="uint8")
        else:
            image_data = None
        for i_image in range(n):
            for i_cam in range(self.ncams):
                if image_data is not None:
                    image_data[:, :, i_image, i_cam] = image
                else:
                    image_data = image
            return image_data


cam = VirtualCam()
cam.acquire_images(n=5)
