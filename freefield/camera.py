import numpy  # for some reason numpy must be imported before PySpin
try:
    import PySpin
except ModuleNotFoundError:
    print("PySpin module required for working with FLIR cams not found! \n"
          "You can download the .whl here: \n"
          "https://www.flir.com/products/spinnaker-sdk/")
import PIL
from freefield import PoseEstimator, DIR
from slab.psychoacoustics import Trialsequence
import time
import cv2
from scipy import stats
from matplotlib import pyplot as plt
import pandas as pd
import logging
import numpy as np
from abc import abstractmethod


class Cameras:
    def __init__(self):
        self.model = PoseEstimator()
        self.calibration = None

    @abstractmethod
    def acquire_images(self) -> None:
        pass

    def test_function(self):
        print("Hello")

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


    def calibrate_camera(self, n_reps=1):
        """Calibrate camera(s) by computing the linear regression for a number of
        points in camera and world coordinates.
        If the camera is a webcam you need to give a list of tuples with
        elevation and azimuth (in that order) of points in your environment.
        You will be asked to point your head towards the target positions in
        randomized order and press enter. If the cameras are the FLIR cameras in
        the freefield the target positions are defined automatically (all speaker
        positions that have LEDs attached). The LEDs will light up in randomized
        order. Point your head towards the lit LED, and press the button on
        the response box. For each position, repeated n times, the headpose will
        be computed. The regression coefficients (slope and intercept) will be
        stored in the environment variables _azi_reg and _ele_reg. The coordinates
        used to compute the regression are also returned (mostly for debugging
        purposes).
        Attributes:
        targets (list of tuples): elevation and azimuth for any number of
            points in world coordinates
        n_repeat (int): number of repetitions for each target (default = 1)

        return: world_coordinates, camera_coordinates (list of tuples)
        """
        # azimuth and elevation of a set of points in camera and world coordinates
        # one list for each camera
        coords = pd.DataFrame(columns=["ele_cam", "azi_ca
                                       "ele_world", "azi_world", "cam", "frame", "n"])
        if _cam_type == "web" and targets is None:
            raise ValueError("Define target positions for calibrating webcam!")
        elif _cam_type == "freefield":
            targets = setup.all_leds()  # get the speakers that have a LED attached
            if setup._mode != "camera_calibration":
                setup.initialize_devices(mode="camera_calibration")
        elif _cam_type is None:
            raise ValueError("Initialize Camera before calibration!")
        if not setup._mode == "camera_calibration":  # initialize setup
            setup.initialize_devices(mode="camera_calibration")
        seq = Trialsequence(n_reps=n_reps, conditions=targets)
        while seq.n_remaining:
            target = seq.__next__()
            if _cam_type == "web":  # look at target position and press enter
                ele, azi = target[0], target[1]
                input("point your head towards the target at elevation: %s and "
                      "azimuth %s. \n Then press enter to take an image an get "
                      "the headpose" % (ele, azi))
            elif _cam_type == "freefield":  # light LED and wait for button press
                ele, azi = target[4], target[3]
                proc, bitval = target[6], target[5]
                setup.printv("trial nr %s: speaker at ele: %s and azi: of %s" %
                             (seq.this_n, ele, azi))
                setup.set_variable(variable="bitmask", value=bitval, proc=proc)
                while not setup.get_variable(variable="response", proc="RP2",
                                             supress_print=True):
                    time.sleep(0.1)  # wait untill button is pressed
            pose = get_headpose(average=False, convert=False, cams=cams)
            pose = pose.rename(columns={"ele": "ele_cam", "azi": "azi_cam"})
            pose.insert(0, "n", seq.this_n)
            pose.insert(2, "ele_world", ele)
            pose.insert(4, "azi_world", azi)
            pose = pose.dropna()
            coords = coords.append(pose, ignore_index=True, sort=True)
        if _cam_type == "freefield":
            setup.set_variable(variable="bitmask", value=0, proc="RX8s")

        camera_to_world(coords)
        return coords

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


def initialize_cameras(kind="flir"):
    if kind.lower() == "flir":
        return FlirCams()
    elif kind.lower() == "webcam":
        return WebCams()


"""

def calibrate_camera(targets=None, n_reps=1, cams="all"):
    Calibrate camera(s) by computing the linear regression for a number of
    points in camera and world coordinates.
    If the camera is a webcam you need to give a list of tuples with
    elevation and azimuth (in that order) of points in your environment.
    You will be asked to point your head towards the target positions in
    randomized order and press enter. If the cameras are the FLIR cameras in
    the freefield the target positions are defined automatically (all speaker
    positions that have LEDs attached). The LEDs will light up in randomized
    order. Point your head towards the lit LED, and press the button on
    the response box. For each position, repeated n times, the headpose will
    be computed. The regression coefficients (slope and intercept) will be
    stored in the environment variables _azi_reg and _ele_reg. The coordinates
    used to compute the regression are also returned (mostly for debugging
    purposes).
    Attributes:
    targets (list of tuples): elevation and azimuth for any number of
        points in world coordinates
    n_repeat (int): number of repetitions for each target (default = 1)

    return: world_coordinates, camera_coordinates (list of tuples)

    # azimuth and elevation of a set of points in camera and world coordinates
    # one list for each camera
    coords = pd.DataFrame(columns=["ele_cam", "azi_ca
                                   "ele_world", "azi_world", "cam", "frame", "n"])
    if _cam_type == "web" and targets is None:
        raise ValueError("Define target positions for calibrating webcam!")
    elif _cam_type == "freefield":
        targets = setup.all_leds()  # get the speakers that have a LED attached
        if setup._mode != "camera_calibration":
            setup.initialize_devices(mode="camera_calibration")
    elif _cam_type is None:
        raise ValueError("Initialize Camera before calibration!")
    if not setup._mode == "camera_calibration":  # initialize setup
        setup.initialize_devices(mode="camera_calibration")
    seq = Trialsequence(n_reps=n_reps, conditions=targets)
    while seq.n_remaining:
        target = seq.__next__()
        if _cam_type == "web":  # look at target position and press enter
            ele, azi = target[0], target[1]
            input("point your head towards the target at elevation: %s and "
                  "azimuth %s. \n Then press enter to take an image an get "
                  "the headpose" % (ele, azi))
        elif _cam_type == "freefield":  # light LED and wait for button press
            ele, azi = target[4], target[3]
            proc, bitval = target[6], target[5]
            setup.printv("trial nr %s: speaker at ele: %s and azi: of %s" %
                         (seq.this_n, ele, azi))
            setup.set_variable(variable="bitmask", value=bitval, proc=proc)
            while not setup.get_variable(variable="response", proc="RP2",
                                         supress_print=True):
                time.sleep(0.1)  # wait untill button is pressed
        pose = get_headpose(average=False, convert=False, cams=cams)
        pose = pose.rename(columns={"ele": "ele_cam", "azi": "azi_cam"})
        pose.insert(0, "n", seq.this_n)
        pose.insert(2, "ele_world", ele)
        pose.insert(4, "azi_world", azi)
        pose = pose.dropna()
        coords = coords.append(pose, ignore_index=True, sort=True)
    if _cam_type == "freefield":
        setup.set_variable(variable="bitmask", value=0, proc="RX8s")

    camera_to_world(coords)
    return coords


def camera_to_world(coords, plot=True):
    Find linear regression for camera and world coordinates and store
    them in global variables
    global _cal
    _cal = pd.DataFrame(columns=["a", "b", "cam", "angle", "min", "max"])
    if plot:
        fig, ax = plt.subplots(2)
        fig.suptitle("World vs Camera Coordinates")

    for cam in pd.unique(coords["cam"]):  # calibrate each camera
        cam_coords = coords[coords["cam"] == cam]
        # find regression coefficients for azimuth and elevation
        for i, angle in enumerate(["ele", "azi"]):
            y = cam_coords[angle+"_cam"].values.astype(float)
            x = cam_coords[angle+"_world"].values.astype(float)
            min, max = cam_coords[angle+"_world"].min(), cam_coords[angle+"_world"].max()
            b, a, r, _, _ = stats.linregress(x, y)
            if np.abs(r) < 0.85:
                setup.printv("For cam %s %s correlation between camera and"
                             "world coordinates is only %s! \n"
                             "There might be something wrong..."
                             % (cam, angle, r))
            _cal = _cal.append(pd.DataFrame([[a, b, cam, angle, min, max]],
                                            columns=["a", "b", "cam", "angle", "min", "max"]),
                               ignore_index=True)
            if plot:
                ax[i].scatter(x, y)
                ax[i].plot(x, x*b+a, linestyle="--", label=cam)
                ax[i].set_title(angle)
                ax[i].legend()
                ax[i].set_xlabel("world coordinates in degree")
                ax[i].set_ylabel("camera coordinates in degree")
    if plot:
        plt.show()


def halt():
    global _cams
    if _cam_type == "freefield":
        for cam in _cams:
            if cam.IsInitialized():
                cam.EndAcquisition()
                cam.DeInit()
            del cam
        _cams.Clear()
        _system.ReleaseInstance()
    if _cam_type == "web":
        for cam in _cams:
            cam.release()
    setup.printv("Deinitializing _camera...")
"""
