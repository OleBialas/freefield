import numpy
from freefield import DIR, Cameras
import cv2
import os
from headpose import PoseEstimator


class VirtualCam(Cameras):
    def __init__(self, n_cams):
        super().__init__()
        self.n_cams = n_cams
        test_image = self.acquire_images(n_images=1)
        self.imsize = test_image.shape[0:2]
        self.model = PoseEstimator()

    def acquire_images(self, n_images):

        image = numpy.random.choice(os.listdir(DIR/"tests"/"images"))
        image = cv2.imread(str(DIR/"tests"/"images"/image))[:, :, 0]
        if self.imsize is not None:
            image_data = numpy.zeros(self.imsize + (n_images, self.n_cams), dtype="uint8")
        else:
            image_data = None
        for i_image in range(n_images):
            for i_cam in range(self.n_cams):
                if image_data is not None:
                    image_data[:, :, i_image, i_cam] = image
                else:
                    return image
        return image_data

    def halt(self):
        pass


def test_image_acquisition():
    for _ in range(10):
        n_cams = numpy.random.randint(1, 5)
        cams = VirtualCam(n_cams=n_cams)
        for _ in range(10):
            n_images = numpy.random.randint(1, 10)
            images = cams.acquire_images(n_images=n_images)
            assert images.shape[0:2] == cams.imsize
            assert images.shape[2] == n_images
            assert images.shape[3] == n_cams


def test_head_pose_acquisition():
    for _ in range(10):
        n_cams = numpy.random.randint(1, 3)
        cams = VirtualCam(n_cams=n_cams)
        n_images = numpy.random.randint(1, 5)
        numpy.testing.assert_raises(ValueError, cams.get_head_pose, **{"convert": True, "n_images": n_images})
        pose = cams.get_head_pose(convert=False, average_axis=None, n_images=n_images)
        assert pose.shape == (2, n_images, n_cams)
        for i in range(cams.n_cams):
            cams.calibration[f"cam{i}"] = {"azimuth": {"a": 0, "b": 1}, "elevation": {"a": 0, "b": 1}}
        pose = cams.get_head_pose(convert=True, average_axis=None, n_images=n_images)
        assert pose.shape == (2, n_images, n_cams)
        pose = cams.get_head_pose(average_axis=1, n_images=n_images)
        assert pose.shape == (2, n_cams)
        pose = cams.get_head_pose(average_axis=(1, 2), n_images=n_images)
        assert pose.shape == (2,)


def test_calibration():
    for _ in range(10):
        n_cams = numpy.random.randint(1, 4)
        cams = VirtualCam(n_cams=n_cams)
        numpy.testing.assert_raises(ValueError, cams.get_head_pose, **{"convert": True})
        n_images = numpy.random.randint(1, 5)
        n = numpy.random.randint(3, 10)  # number of points for the calibration
        camera_coordinates = [cams.get_head_pose(convert=False, n_images=n_images, average_axis=1) for __ in range(n)]
        factor = numpy.random.uniform(-1, 1)
        offset = numpy.random.uniform(-10, 10)
        world_coordinates = [c[:, 0]*factor+offset for c in camera_coordinates]
        cams.calibrate(world_coordinates, camera_coordinates, plot=False)
        for angle in ["azimuth", "elevation"]:
            a, b = cams.calibration["cam0"][angle].values()
            numpy.testing.assert_almost_equal([a, b], [offset, factor])
        for cam in range(n_cams):
            assert "azimuth" in cams.calibration[f"cam{cam}"]
            assert "elevation" in cams.calibration[f"cam{cam}"]
            for angle in ["azimuth", "elevation"]:
                assert "a" in cams.calibration[f"cam{cam}"][angle]
                assert "b" in cams.calibration[f"cam{cam}"][angle]
