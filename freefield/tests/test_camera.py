import numpy
from freefield import DIR, Cameras
import cv2
import os
from freefield.headpose import PoseEstimator
import pandas as pd
import numpy as np


class VirtualCam(Cameras):
    def __init__(self):
        super().__init__()
        self.ncams = 1
        self.imsize = self.acquire_images(n=1).shape[0:2]

    def acquire_images(self, n=1):

        image = numpy.random.choice(os.listdir(DIR/"tests"/"images"))
        image = cv2.imread(str(DIR/"tests"/"images"/image))[:, :, 0]
        if hasattr(self, "imsize"):
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


def test_camera():
    cam = VirtualCam()
    assert hasattr(cam, "imsize")
    pose = cam.get_headpose(convert=False, average=False, n=5, resolution=.8)
    assert len(pose) == 5


def test_calibration():
    cam = VirtualCam()
    coords = pd.read_csv(DIR/"tests"/"coordinates.csv")
    cam.calibrate(coords)
    pose = cam.get_headpose(convert=False, average=False, n=5, resolution=.8)
    assert len(pose) == 5
    assert all(coords.frame == "camera")
    pose = cam.get_headpose(convert=True, average=True, n=5, resolution=.8)
    assert len(pose) == 1
    assert all(coords.frame == "world")


def test_headpose():
    imfolder = DIR/"tests"/"images"
    images = [im for im in os.listdir(imfolder) if "jpg" in im]
    df = pd.read_csv(DIR/"tests"/"pose.csv")
    df = df.where(pd.notnull(df), None)
    for thresh in [0.3, 0.9, 0.99]:
        pose_estimator = PoseEstimator(threshold=thresh)
        for image in images:
            img = cv2.imread(str(DIR/"tests"/"images"/image))
            azi, ele = pose_estimator.pose_from_image(img)
            row = df[(df['image'] == image) & (df['threshold'] == thresh)]
            if azi is not None and ele is not None:
                assert azi.round(3) == np.float64(row["azimuth"]).round(3)
                assert ele.round(3) == np.float64(row["elevation"]).round(3)
            else:
                assert azi == row["azimuth"].values[0]
                assert ele == row["elevation"].values[0]


camera = VirtualCam()
coords = pd.read_csv(DIR/"tests"/"coordinates.csv")
camera.calibrate(coords)
coords = camera.get_headpose(convert=False, average=False, n=5, resolution=.8)

for cam in np.unique(coords["cam"]):  # convert for each cam ...
    for angle in ["azi", "ele"]:  # ... and each angle
        print(cam, angle)
        reg = camera.calibration[np.logical_and(camera.calibration["cam"] == cam, camera.calibration["angle"] == angle)]
        a, b = reg["a"].values[0], reg["b"].values[0]
        coords.loc[coords["cam"] == cam, angle] = \
            (coords[coords["cam"] == cam][angle] - a) / b
        coords.insert(3, "frame", "world")
