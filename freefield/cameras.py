import time
import logging
from abc import abstractmethod
import numpy
from matplotlib import pyplot as plt
from scipy import stats
import PIL
import cv2
try:
    import PySpin
except ModuleNotFoundError:
    PySpin = False
try:
    import headpose
except ModuleNotFoundError:
    headpose = False

def initialize(kind="flir"):
    """
    Initialize connected cameras for head pose estimation.
    Arguments:
        kind (str): The type of camera used. If "flir", use the PySpin API to operate the cameras,
            if "webcam" use opencv.
    Returns:
        (instance of Cameras): Object for handling the initialized cameras.
    """
    if kind.lower() == "flir":
        return FlirCams()
    elif kind.lower() == "webcam":
        return WebCams()
    else:
        raise ValueError("Possible camera types are 'flir' or 'webcam'")

class Cameras:
    def __init__(self):
        self.imsize = None
        self.model = None
        self.calibration = dict()
        self.n_cams = None

    def set_detection_threshold(self, threshold):
        self.model.threshold = threshold

    @abstractmethod
    def acquire_images(self, n_images):
        pass

    @abstractmethod
    def halt(self):
        pass

    def get_head_pose(self, convert=True, average_axis=1, n_images=1, resolution=1.0):
        """Acquire n images and compute head pose (elevation and azimuth). If
        convert is True use the regression coefficients to convert
        the camera into world coordinates. """
        if self.model is None:
            raise ImportError("Headpose estimation requires the headpose module (pip install headpose)!")
        images = self.acquire_images(n_images)  # take images
        pose = numpy.zeros([2, images.shape[2], images.shape[3]])
        for i_cam in range(images.shape[3]):
            for i_image in range(images.shape[2]):
                image = images[:, :, i_image, i_cam]  # get image from array
                if resolution < 1.0:
                    image = self.change_image_res(image, resolution)
                azimuth, elevation, _ = self.model.pose_from_image(image)
                pose[:, i_image, i_cam] = azimuth, elevation
        if convert:
            pose = self.convert_coordinates(pose)
        if average_axis is not None:
            pose = pose.mean(axis=average_axis)
        return pose

    def change_image_res(self, image, resolution):
        image = PIL.Image.fromarray(image)
        width = int(self.imsize[1] * resolution)
        height = int(self.imsize[0] * resolution)
        image = image.resize((width, height), PIL.Image.ANTIALIAS)
        return numpy.asarray(image)

    def convert_coordinates(self, pose):
        if not self.calibration:
            raise ValueError("Camera has to be calibrated to convert coordinates")
        for i_cam in range(pose.shape[2]):  # convert all images for each cam ...
            for i_angle, angle in enumerate(["azimuth", "elevation"]):  # ... and each angle
                a, b = self.calibration[f"cam{i_cam}"][angle]["a"], self.calibration[f"cam{i_cam}"][angle]["b"]
                pose[i_angle, :, i_cam] = a + b * pose[i_angle, :, i_cam]
        return pose

    def calibrate(self, world_coordinates, camera_coordinates, plot=True):
        if not len(world_coordinates) == len(camera_coordinates):
            raise ValueError("Camera and world coordinates must be of the same shape!")
        if plot is True:
            fig, ax = plt.subplots(2)
            fig.suptitle("World vs Camera Coordinates")
        for i_cam in range(self.n_cams):  # calibrate each camera
            self.calibration[f"cam{i_cam}"] = {"azimuth": {}, "elevation": {}}
            for i_angle, angle in enumerate(["azimuth", "elevation"]):  # ... and each angle
                x = [c[i_angle, i_cam] for c in camera_coordinates]
                y = [w[i_angle] for w in world_coordinates]
                b, a, r, _, _ = stats.linregress(x, y)
                if numpy.abs(r) < 0.85:
                    logging.warning(f"Correlation for camera {i_cam} {angle} is only {r}!")
                self.calibration[f"cam{i_cam}"][angle] = {"a": a, "b": b}
                if plot is True:
                    ax[i_angle].scatter(x, y)
                    ax[i_angle].plot(x, numpy.array(x) * b + a, linestyle="--", label=f"cam{i_cam}")
                    ax[i_angle].set_title(angle)
                    ax[i_angle].legend()
                    ax[i_angle].set_xlabel("camera coordinates in degree")
                    ax[i_angle].set_ylabel("world coordinates in degree")
        if plot:
            plt.show()


class FlirCams(Cameras):
    def __init__(self):
        if PySpin is False:
            raise ValueError("PySpin module required for working with FLIR cams not found! \n"
                             "You can download the .whl here: \n"
                             "https://www.flir.com/products/spinnaker-sdk/")
        super().__init__()
        if headpose is False:
            self.model = None
        else:
            self.model = headpose.PoseEstimator()
        self.system = PySpin.System.GetInstance()
        self.cams = self.system.GetCameras()
        self.n_cams = self.cams.GetSize()
        if self.n_cams == 0:  # Finish if there are no cameras
            self.cams.Clear()  # Clear camera list before releasing system
            self.system.ReleaseInstance()  # Release system instance
            logging.warning('No camera found!')
        else:
            for cam in self.cams:
                cam.Init()  # Initialize camera
            logging.info(f"initialized {self.n_cams} FLIR camera(s)")
        imsize = self.acquire_images(n_images=1).shape[0:2]
        self.imsize = imsize

    def acquire_images(self, n_images=1):
        # TODO: ideas to make this faster -> only set nodemap once use async
        if self.imsize is not None:
            image_data = numpy.zeros(self.imsize + (n_images, self.n_cams), dtype="uint8")
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
            acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()
            node_acquisition_mode.SetIntValue(acquisition_mode_continuous)
            cam.BeginAcquisition()
        for i_image in range(n_images):
            for i_cam, cam in enumerate(self.cams):
                time.sleep(0.1)
                image_result = cam.GetNextImage()
                if image_result.IsIncomplete():
                    raise ValueError(f'Image incomplete: image status'
                                     '{image_result.GetImageStatus()}')
                image = image_result.Convert(
                    PySpin.PixelFormat_Mono8, PySpin.HQ_LINEAR)
                image = image.GetNDArray()
                image.setflags(write=1)
                image_result.Release()
                if image_data is not None:
                    image_data[:, :, i_image, i_cam] = image
                else:
                    _ = [cam.EndAcquisition() for cam in self.cams]
                    return image
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
        super().__init__()
        if headpose is False:
            self.model = None
        else:
            self.model = headpose.PoseEstimator()
        self.cams = []
        stop = False
        while stop is False:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                self.cams.append(cap)
            else:
                stop = True
        logging.info("initialized %s webcams(s)" % (len(self.cams)))
        self.n_cams = len(self.cams)
        self.imsize = self.acquire_images(n_images=1).shape[0:2]

    def acquire_images(self, n_images=1):
        """
            The webcam takes several pictures and reading only advances
            the buffer one step at a time, thus grab all images and only
            retrieve the latest one
        """
        if self.imsize is not None:
            image_data = numpy.zeros(self.imsize + (n_images, self.n_cams), dtype="uint8")
        else:
            image_data = None
        for i_image in range(n_images):
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
                        return image
        return image_data

    def halt(self):
        for cam in self.cams:
            cam.release()
        logging.info("Halting webcams.")
