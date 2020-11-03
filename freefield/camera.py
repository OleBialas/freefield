import numpy  # for some reason numpy must be imported before PySpin
try:
    import PySpin
except ModuleNotFoundError:
    print("PySpin module required for working with FLIR cams not found! \n"
          "You can download the .whl here: \n"
          "https://www.flir.com/products/spinnaker-sdk/")
import PIL
from freefield import PoseEstimator
import time
import cv2
from scipy import stats
from matplotlib import pyplot as plt
import pandas as pd
import logging
import numpy as np
from abc import abstractmethod


def initialize_cameras(kind="flir"):
    if kind.lower() == "flir":
        return FlirCams()
    elif kind.lower() == "webcam":
        return WebCams()


class Cameras:
    def __init__(self):
        self.model = PoseEstimator()
        self.calibration = None

    @abstractmethod
    def acquire_images(self) -> None:
        pass

    @abstractmethod
    def halt(self) -> None:
        pass

    def get_headpose(self, convert=True, average=True, n=1, resolution=1.0):
        """Acquire n images and compute headpose (elevation and azimuth). If
        convert is True use the regression coefficients to convert
        the camera into world coordinates
        """
        pose = pd.DataFrame(columns=["ele", "azi", "cam"])
        images = self.acquire_image(n)  # take images
        for i_cam in range(images.shape[3]):
            for i_image in range(images.shape[2]):
                image = images[:, :, i_image, i_cam]  # get image from array
                image = self.change_image_res(image, resolution)
                # get the headpose,
                ele, azi, _ = self.model.pose_from_image(numpy.asarray(image))
                pose = pose.append(
                        pd.DataFrame(
                            [[ele, azi, i_cam, "world"]],
                            columns=["ele", "azi", "cam", "frame"]))
        if len(pose.dropna()) == 0:
            return pose  # if all are NaN, no face was found in any image
        if convert and (self.calibration is not None):
            pose = self.convert_coordinates(pose)
            if average:  # only return the mean
                return pose.ele.mean(), pose.azi.mean()
            else:  # return the whole data frame
                return pose
        else:
            logging.warning("Camera is not calibrated!")

    def change_image_res(self, image, resolution):
        width = height = int(self.imsize[1]*resolution)
        image = image.resize((width, height), PIL.Image.ANTIALIAS)
        image = PIL.Image.fromarray(image)
        return image

    def convert_coordinates(self, coords):
        for cam in np.unique(coords["cam"]):  # convert for each cam ...
            for angle in ["azi", "ele"]:  # ... and each angle
                reg = self.calibration[(self.calibration["cam"] == cam)
                                       (self.calibration["angle"] == angle)]

                coords.loc[coords["cam"] == cam, angle] = \
                    ((coords[coords["cam"] == cam][angle] - reg["a"].values)
                        / reg["b"].values)
                coords.insert(3, "frame", "world")

    def calibrate(self, coords, plot=True):
        calibration = pd.DataFrame(columns=["a", "b", "cam", "angle"])
        if plot:
            fig, ax = plt.subplots(2)
            fig.suptitle("World vs Camera Coordinates")

        for cam in pd.unique(coords["cam"]):  # calibrate each camera
            cam_coords = coords[coords["cam"] == cam]
            # find regression coefficients for azimuth and elevation
            for i, angle in enumerate(["ele", "azi"]):
                y = cam_coords[angle+"_cam"].values.astype(float)
                x = cam_coords[angle+"_world"].values.astype(float)
                b, a, r, _, _ = stats.linregress(x, y)
                if np.abs(r) < 0.85:
                    logging.warning(f"Correlation for camera {cam}",
                                    f"{angle} is only {r}!")
                row = {"a": a, "b": b, "cam": cam, "angle": angle}
                calibration = calibration.append(row, ignore_index=True)
                if plot:
                    ax[i].scatter(x, y)
                    ax[i].plot(x, x*b+a, linestyle="--", label=cam)
                    ax[i].set_title(angle)
                    ax[i].legend()
                    ax[i].set_xlabel("world coordinates in degree")
                    ax[i].set_ylabel("camera coordinates in degree")

        self.calibration = calibration
        if plot:
            plt.show()


class FlirCams(Cameras):
    def __init__(self):
        super().__init__()
        self.system = PySpin.System.GetInstance()
        self.cams = self._system.GetCameras()
        self.ncams = self.cams.GetSize()
        self.imsize = self.acquire_images(n=1).shape[0:2]
        if self.ncams == 0:    # Finish if there are no cameras
            self.cams.Clear()  # Clear camera list before releasing system
            self._system.ReleaseInstance()  # Release system instance
            logging.warning('No camera found!')
        else:
            for cam in self.cams:
                cam.Init()  # Initialize camera
            logging.info(f"initialized {self.ncams} FLIR camera(s)")

    def acquire_images(self, n=1):
        if hasattr(self, "imagesize"):
            image_data = np.zeros((self.imsize)+(n, self.ncams), dtype="uint8")
        else:
            image_data = None
        for cam in self.cams:  # start the cameras
            node_acquisition_mode = PySpin.CEnumerationPtr(
                cam.GetNodeMap().GetNode('AcquisitionMode'))
            if (not PySpin.IsAvailable(node_acquisition_mode) or
                    not PySpin.IsWritable(node_acquisition_mode)):
                raise ValueError(
                    'Unable to set acquisition to continuous, aborting...')
            node_acquisition_mode_continuous = \
                node_acquisition_mode.GetEntryByName('Continuous')
            if (not PySpin.IsAvailable(node_acquisition_mode_continuous) or
                    not PySpin.IsReadable(node_acquisition_mode_continuous)):
                raise ValueError(
                    'Unable to set acquisition to continuous, aborting...')
            acquisition_mode_continuous = \
                node_acquisition_mode_continuous.GetValue()
            node_acquisition_mode.SetIntValue(acquisition_mode_continuous)
            cam.BeginAcquisition()
        for i_image in range(n):
            for i_cam, cam in enumerate(self.cams):
                time.sleep(0.01)
                image_result = cam.GetNextImage()
                if image_result.IsIncomplete():
                    raise ValueError('Image incomplete: image status %d ...'
                                     % image_result.GetImageStatus())
                image = image_result.Convert(
                    PySpin.PixelFormat_Mono8, PySpin.HQ_LINEAR)
                image = image.GetNDArray()
                image_result.Release()
                if image_data is not None:
                    image_data[:, :, i_image, i_cam] = image
                else:
                    image_data = image
        return image_data

    def halt(self):
        for cam in self.cams:
            if cam.IsInitialized():
                cam.EndAcquisition()
                cam.DeInit()
            del cam
        self.cams.Clear()
        self.system.ReleaseInstance()
        logging.info("Halting FLIR cameras.")


class WebCams(Cameras):
    def __init__(self):
        super().__init__()
        self.cams = []
        stop = False
        while stop is False:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                self.cams.append(cap)
            else:
                stop = True
        logging.info("initialized %s webcams(s)" % (len(self.cams)))
        self.ncams = len(self.cams)
        self.imsize = self.acquire_images(n=1).shape[0:2]

    def acquire_images(self, n=1):
        """
            The webcam takes several pictures and reading only advances
            the buffer one step at a time, thus grab all images and only
            retrieve the latest one
        """
        if hasattr(self, "imagesize"):
            image_data = np.zeros((self.imsize)+(n, self.ncams), dtype="uint8")
        else:
            image_data = None
        for i_image in range(n):
            for i_cam, cam in enumerate(self.cams):
                for i in range(cv2.CAP_PROP_FRAME_COUNT):
                    cam.grab()
                ret, image = cam.retrieve()
                if ret is False:
                    logging.warning("could not acquire image...")
            if image_data is not None:
                image_data[:, :, i_image, i_cam] = image
            else:
                image_data = image
        return image_data

    def halt(self):
        for cam in self.cams:
            cam.release()
        logging.info("Halting webcams.")
