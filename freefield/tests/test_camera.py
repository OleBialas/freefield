import numpy as np
from freefield import DIR, Cameras
import cv2
import os
import pandas as pd


class VirtualCam(Cameras):
    def __init__(self):
        super().__init__()
        self.ncams = 1
        test_image = self.acquire_images(n=1)
        self.imsize = test_image.shape[0:2]

    def acquire_images(self, n: int = 1) -> np.ndarray:

        image = np.random.choice(os.listdir(DIR/"tests"/"images"))
        image = cv2.imread(str(DIR/"tests"/"images"/image))[:, :, 0]
        if hasattr(self, "imsize"):
            image_data = np.zeros(self.imsize+(n, self.ncams), dtype="uint8")
        else:
            image_data = None
        for i_image in range(n):
            for i_cam in range(self.ncams):
                if image_data is not None:
                    image_data[:, :, i_image, i_cam] = image
                else:
                    image_data = image
        return image_data

    def halt(self):
        pass


def test_camera():
    cam = VirtualCam()
    assert hasattr(cam, "imsize")
    pose = cam.get_headpose(convert=False, average=False, n=5, resolution=.8)
    assert len(pose) == 5


def test_calibration():
    cam = VirtualCam()
    pose = cam.get_headpose(convert=True, average=False, n=5, resolution=.8)
    assert pose is None
    coords = pd.read_csv(DIR/"tests"/"coordinates.csv")
    cam.calibrate(coords)
    pose = cam.get_headpose(convert=False, average=False, n=5, resolution=.8)
    assert len(pose) == 5 and isinstance(pose, pd.core.frame.DataFrame)
    assert all(pose.frame == "camera")
    pose = cam.get_headpose(convert=True, average=False, n=1, resolution=.8)
    assert all(pose.frame == "world")
    pose = cam.get_headpose(convert=True, average=True, n=5, resolution=.8)
    assert len(pose) == 2 and isinstance(pose, tuple)
