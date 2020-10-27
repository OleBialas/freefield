from freefield import DATADIR
from freefield.headpose import PoseEstimator
import cv2
import os
import pandas as pd
imfolder = DATADIR/"test_images"
images = [im for im in os.listdir(imfolder) if "jpg" in im]

df = pd.DataFrame(columns=["image", "threshold", "azimuth", "elevation"])

for thresh in [0.3, 0.9, 0.99]:
    pose_estimator = PoseEstimator(threshold=thresh)
    for image in images:
        img = cv2.imread(str(DATADIR/"test_images"/image))
        azi, ele = pose_estimator.pose_from_image(img)
        row = dict(image=image, threshold=thresh, azimuth=azi, elevation=ele)
        df = df.append(row, ignore_index=True)
        df.to_csv(DATADIR/"test_images"/"test_pose.csv")


def test_headpose():
    pass
