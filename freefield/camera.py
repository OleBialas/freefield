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


class Cameras():
    def __init__(self, face_detection_tresh=.9):
        self.model = PoseEstimator(threshold = face_detection_tresh)
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
        images = self.acquire_images(n)  # take images
        for i_cam in range(images.shape[3]):
            for i_image in range(images.shape[2]):
                image = images[:, :, i_image, i_cam]  # get image from array
                if resolution < 1.0:
                    image = self.change_image_res(image, resolution)
                # get the headpose,
                azi, ele = self.model.pose_from_image(image)
                pose = pose.append(
                        pd.DataFrame(
                            [[ele, azi, i_cam, "camera"]],
                            columns=["ele", "azi", "cam", "frame"]))
        if len(pose.dropna()) == 0:
            if average:
                return (None, None)
            else:
                return pose
        if convert:
            if self.calibration is None:
                logging.warning("Camera is not calibrated!")
                return None
            else:
                pose = self.convert_coordinates(pose)
        if average:  # only return the mean
            return pose.azi.mean(), pose.ele.mean()
        else:  # return the whole data frame
            return pose

    def change_image_res(self, image, resolution):
        image = PIL.Image.fromarray(image)
        width = int(self.imsize[1]*resolution)
        height = int(self.imsize[0]*resolution)
        image = image.resize((width, height), PIL.Image.ANTIALIAS)
        return numpy.asarray(image)

    def convert_coordinates(self, coords):
        for cam in np.unique(coords["cam"]):  # convert for each cam ...
            for angle in ["azi", "ele"]:  # ... and each angle
                # get the regression coefficients a & b
                reg = self.calibration[np.logical_and(
                    self.calibration["cam"] == cam,
                    self.calibration["angle"] == angle)]
                a, b = reg["a"].values[0], reg["b"].values[0]
                coords.loc[coords["cam"] == cam, angle] = \
                    a + b * coords[coords["cam"] == cam][angle]
        coords.frame = "world"  # coords are now in "world" frame
        return coords

    def calibrate(self, coords, plot=True):
        calibration = pd.DataFrame(columns=["a", "b", "cam", "angle"])
        if plot:
            fig, ax = plt.subplots(2)
            fig.suptitle("World vs Camera Coordinates")
        for cam in pd.unique(coords["cam"]):  # calibrate each camera
            cam_coords = coords[coords["cam"] == cam]
            # find regression coefficients for azimuth and elevation
            for i, angle in enumerate(["ele", "azi"]):
                x = cam_coords[angle+"_cam"].values.astype(float)
                y = cam_coords[angle+"_world"].values.astype(float)
                b, a, r, _, _ = stats.linregress(x, y)
                if np.abs(r) < 0.85:
                    logging.warning(f"Correlation for camera {cam} {angle} is only {r}!")
                row = {"a": a, "b": b, "cam": cam, "angle": angle}
                calibration = calibration.append(row, ignore_index=True)
                if plot:
                    ax[i].scatter(x, y)
                    ax[i].plot(x, x*b+a, linestyle="--", label=cam)
                    ax[i].set_title(angle)
                    ax[i].legend()
                    ax[i].set_xlabel("camera coordinates in degree")
                    ax[i].set_ylabel("world coordinates in degree")
        self.calibration = calibration
        if plot:
            plt.show()


class FlirCams(Cameras):
    def __init__(self, face_detection_tresh=.9):
        super().__init__(face_detection_tresh=face_detection_tresh)
        self.system = PySpin.System.GetInstance()
        self.cams = self.system.GetCameras()
        self.ncams = self.cams.GetSize()
        if self.ncams == 0:    # Finish if there are no cameras
            self.cams.Clear()  # Clear camera list before releasing system
            self.system.ReleaseInstance()  # Release system instance
            logging.warning('No camera found!')
        else:
            for cam in self.cams:
                cam.Init()  # Initialize camera
            logging.info(f"initialized {self.ncams} FLIR camera(s)")
        imsize = self.acquire_images(n=1).shape[0:2]
        self.imsize = imsize

    def acquire_images(self, n=1):
        # TODO: ideas to make this faster -> only set nodemap once use async
        if hasattr(self, "imsize"):
            image_data = np.zeros(self.imsize+(n, self.ncams), dtype="uint8")
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
            acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()
            node_acquisition_mode.SetIntValue(acquisition_mode_continuous)
            cam.BeginAcquisition()
        for i_image in range(n):
            for i_cam, cam in enumerate(self.cams):
                time.sleep(0.1)
                image_result = cam.GetNextImage()
                if image_result.IsIncomplete():
                    raise ValueError('Image incomplete: image status %d ...'
                                     % image_result.GetImageStatus())
                image = image_result.Convert(
                    PySpin.PixelFormat_Mono8, PySpin.HQ_LINEAR)
                image = image.GetNDArray()
                image.setflags(write=1)
                image_result.Release()
                if hasattr(self, "imsize"):
                    image_data[:, :, i_image, i_cam] = image
                else:
                    image_data = image
        [cam.EndAcquisition() for cam in self.cams]
        return image_data

    def halt(self):
        for cam in self.cams:
            if cam.IsInitialized():
                cam.DeInit()
            del cam
        self.cams.Clear()
        self.system.ReleaseInstance()
        logging.info("Halting FLIR cameras.")


class WebCams(Cameras):
    def __init__(self):
        super().__init__(face_detection_tresh=face_detection_tresh)
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
        if hasattr(self, "imsize"):
            image_data = np.zeros(self.imsize+(n, self.ncams), dtype="uint8")
        else:
            image_data = None
        for i_image in range(n):
            for i_cam, cam in enumerate(self.cams):
                for i in range(cv2.CAP_PROP_FRAME_COUNT):
                    cam.grab()
                ret, image = cam.retrieve()
                if image.shape[-1] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                if ret is False:
                    logging.warning("could not acquire image...")
                else:
                    if image_data is not None:
                        image_data[:, :, i_image, i_cam] = image
                    else:
                        image_data = image
        return image_data

    def halt(self):
        for cam in self.cams:
            cam.release()
        logging.info("Halting webcams.")
