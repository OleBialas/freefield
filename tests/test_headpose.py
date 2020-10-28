from freefield import TESTDIR
from freefield.headpose import PoseEstimator
import cv2
import os
import pandas as pd
import numpy as np


def test_headpose():
    imfolder = TESTDIR/"images"
    images = [im for im in os.listdir(imfolder) if "jpg" in im]
    df = pd.read_csv(TESTDIR/"images"/"pose.csv")
    df = df.where(pd.notnull(df), None)
    for thresh in [0.3, 0.9, 0.99]:
        pose_estimator = PoseEstimator(threshold=thresh)
        for image in images:
            img = cv2.imread(str(TESTDIR/"images"/image))
            azi, ele = pose_estimator.pose_from_image(img)
            row = df[(df['image'] == image) & (df['threshold'] == thresh)]
            if azi is not None and ele is not None:
                assert azi.round(3) == np.float64(row["azimuth"]).round(3)
                assert ele.round(3) == np.float64(row["elevation"]).round(3)
            else:
                assert azi == row["azimuth"].values[0]
                assert ele == row["elevation"].values[0]
