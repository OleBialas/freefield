import numpy  # for some reason numpy must be imported before PySpin
try:
    import PySpin
except ModuleNotFoundError:
    print("PySpin module required for working with FLIR cams not found! \n"
          "You can download the .whl here: \n"
          "https://www.flir.com/products/spinnaker-sdk/")
from pathlib import Path
from freefield import setup
from slab.psychoacoustics import Trialsequence
import time
import cv2
import matplotlib
from scipy import stats
from matplotlib import pyplot as plt
import pandas as pd
import logging
import numpy as np


class Cameras:
    def __init__(self, kind="flir"):
        if kind.lower() == "flir":
            self.cams = initialize_flir()
        self.ncams = 0
        self.cams = None

    def acquire_images(self, n=1):
    image_data = np.zeros((_imagesize)+(n_images, len(cams)), dtype="uint8")


def initialize_flir():
    global _system
    _system = PySpin.System.GetInstance()
    cams = _system.GetCameras()
    ncams = cams.GetSize()
    if ncams == 0:    # Finish if there are no cameras
        cams.Clear()  # Clear camera list before releasing system
        _system.ReleaseInstance()  # Release system instance
        logging.warning('No camera found!')
    else:
        for cam in cams:
            cam.Init()  # Initialize camera
        logging.info(f"initialized {len(cams)} FLIR camera(s)")
    return cams


def initialize_webcam():
    cams, stop, i = [], False, 0
    while stop is False:
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cams.append(cap)
        else:
            stop = True
    logging.info("initialized %s webcams(s)" % (len(_cams)))
    return cams

# get a single image to get the size and estimate camera coefficients
_imagesize = acquire_image(cams=0, n_images=1).shape[0:2]



def acquire_image(cams="all", n_images=1):
    """
    acquire an image from a camera. The camera at the index given by cam
    idx in the camera list _cams is used. If only one camera is being used
    the default cam_idx=0 is sufficient.
    """
    if cams == "all":
        cams = _cams  # use all cameras
    elif isinstance(cams, int):
        cams = [_cams[cams]]
    elif isinstance(cams, list) and all(isinstance(c, int) for c in cams):
        cams = _cams[cams]
    else:
        raise ValueError("cams must be int, list of ints or 'all'!")
    if _cam_type is None:
        raise ValueError("Cameras must be initialized before acquisition")
    if _imagesize is not None:  # ignore when taking image while initializing
        image_data = np.zeros((_imagesize)+(n_images, len(cams)), dtype="uint8")
    if _cam_type == "freefield":  # start the cameras
        for cam in cams:
            if _cam_type == "freefield":
                node_acquisition_mode = PySpin.CEnumerationPtr(
                    cam.GetNodeMap().GetNode('AcquisitionMode'))
                if (not PySpin.IsAvailable(node_acquisition_mode) or
                        not PySpin.IsWritable(node_acquisition_mode)):
                    raise ValueError(
                        'Unable to set acquisition to continuous, aborting...')
                node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName(
                    'Continuous')
                if (not PySpin.IsAvailable(node_acquisition_mode_continuous) or
                        not PySpin.IsReadable(
                        node_acquisition_mode_continuous)):
                    raise ValueError(
                        'Unable to set acquisition to continuous, aborting...')
                acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()
                node_acquisition_mode.SetIntValue(acquisition_mode_continuous)
                cam.BeginAcquisition()
    for i_im in range(n_images):
        for i_cam, cam in enumerate(cams):
            if _cam_type == "freefield":
                time.sleep(0.01)
                image_result = cam.GetNextImage()
                if image_result.IsIncomplete():
                    raise ValueError('Image incomplete: image status %d ...'
                                     % image_result.GetImageStatus())
                image = image_result.Convert(
                    PySpin.PixelFormat_Mono8, PySpin.HQ_LINEAR)
                image = image.GetNDArray()
                image_result.Release()

            elif _cam_type == "web":
                # The webcam takes several pictures and reading only advances
                # the buffer one step at a time, thus grab all images and only
                # retrieve the latest one
                for i in range(cv2.CAP_PROP_FRAME_COUNT):
                    cam.grab()
                ret, image = cam.retrieve()
                if ret is False:
                    setup.printv("could not acquire image, returning None...")
            if _imagesize is not None:
                image_data[:, :, i_im, i_cam] = image
            else:
                image_data = image
    if _cam_type == "freefield":
        [cam.EndAcquisition() for cam in cams]
    return image_data


def set_imagesize(height, width):
    global _imagesize
    for cam in _cams:
        nodemap = cam.GetNodeMap()  # get nodemap
        # set image width:
        node_width = PySpin.CIntegerPtr(nodemap.GetNode('Width'))
        if PySpin.IsAvailable(node_width) and PySpin.IsWritable(node_width):
            if width > node_width.GetMax():
                setup.printv("Can not set width to %s because maximum is %s" %
                             (width, node_width.GetMax()))
            else:
                node_width.SetValue(width)
                imwidth = width
                setup.printv("set width to %s" % (width))
        else:
            print('Width not available...')
        # set image height:
        node_height = PySpin.CIntegerPtr(nodemap.GetNode('Height'))
        if PySpin.IsAvailable(node_height) and PySpin.IsWritable(node_height):
            if height > node_height.GetMax():
                setup.printv("Can not set height to %s because maximum is %s" %
                             (height, node_height.GetMax()))
            else:
                node_height.SetValue(height)
                imheight = height
                setup.printv("set height to %s" % (height))
        else:
            print('Height not available...')
        _imagesize = (imheight, imwidth)


def get_headpose(cams="all", target=(None, None), convert=False, average=False, n_images=None):
    """
    Acquire n images and compute headpose (elevation and azimuth). If
    convert is True use the regression coefficients to convert
    the camera into world coordinates
    """
    # TODO: sanity check the resulting dataframe, e.g. how big is the max diff
    # TODO: downscaling images after recording should be replaced with setting camera resolution
    pose = pd.DataFrame(columns=["ele", "azi", "cam"])
    if n_images is None:  # use default is not specified
        n_images = _nim
    images = acquire_image(cams, n_images)  # take images
    for i_cam in range(images.shape[3]):
        for i_image in range(images.shape[2]):
            image = PIL.Image.fromarray(images[:, :, i_image, i_cam])
            width, height = int(_imagesize[1]*_res), int(_imagesize[0]*_res)
            image = image.resize((width, height), PIL.Image.ANTIALIAS)
            ele, azi, _ = _pose_from_image(numpy.asarray(image))
            row = pd.DataFrame([[ele, azi, i_cam]],
                               columns=["ele", "azi", "cam"])
            pose = pose.append(row)
    if convert:  # convert azimuth and elevation to world coordinates
        if len(pose.dropna()) > 0:
            if _cal is None:
                raise ValueError("Can't convert coordinates because camera is"
                                 "not calibrated!")
            else:
                for cam in np.unique(pose["cam"]):
                    for angle, expected in zip(["azi", "ele"], target):
                        reg = _cal[(_cal["cam"] == cam) & (_cal["angle"] == angle)]
                        if expected is not None:  # only use cam if traget is in range
                            if not reg["min"].values[0] <= expected <= reg["max"].values[0]:
                                pose.loc[pose["cam"] == cam, angle] = np.nan
                        pose.loc[pose["cam"] == cam, angle] = (pose[pose["cam"] == cam][angle] -
                                                               reg["a"].values)/reg["b"].values
            pose.insert(3, "frame", "world")
        else:
            pose.insert(3, "frame", "camera")
    if average:  # only return the mean
        if not convert:
            raise ValueError("Can only average after converting coordinates!")
        return pose.ele.mean(), pose.azi.mean()
    else:  # return the whole data frame
        return pose


def _pose_from_image(image, plot_arg=None):
    """
    Compute the head pose from an image, which must be a 2d (grayscale) or
    3d (color) numpy array. If only_euler=True (default), only angles azimuth,
    elevation and tilt are returned. If False also face shape, rotation vector,
    translation vector, camera matrix , and distortion are returned. This is
    nesseccary to plot the image with the computed headpose.
    """
    focal_length, center = image.shape[1], (image.shape[1]/2, image.shape[0]/2)
    mtx = numpy.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double")
    dist = numpy.zeros((4, 1))  # assuming no lens distortion
    face_rects = _detector(image, 0)  # find the face
    if not face_rects:
        setup.printv("Could not recognize a face in the image, returning none")
        return None, None, None
    shape = _predictor(image, face_rects[0])
    shape = face_utils.shape_to_np(shape)
    image_pts = numpy.float32([shape[17], shape[21], shape[22], shape[26],
                               shape[36], shape[39], shape[42], shape[45],
                               shape[31], shape[35], shape[48], shape[54],
                               shape[57], shape[8]])
    # estimate the translation and rotation coefficients:
    success, rotation_vec, translation_vec = cv2.solvePnP(_object_pts, image_pts, mtx, dist)
    # get the angles out of the transformation matrix:
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, angles = cv2.decomposeProjectionMatrix(pose_mat)
    angles[0, 0] = angles[0, 0]*-1
    if plot_arg is None:
        # elevation, azimuth and tilt
        return angles[0, 0], angles[1, 0], angles[2, 0]
    else:
        reprojectdst, _ = cv2.projectPoints(_reprojectsrc, rotation_vec,
                                            translation_vec, mtx, dist)
        reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

        for (x, y) in shape:
            image = image.astype(np.float32)
            cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
            cv2.putText(image, "Elevation: " + "{:7.2f}".format(angles[0, 0]),
                        (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),
                        thickness=2)
            cv2.putText(image, "Azimuth: " + "{:7.2f}".format(angles[1, 0]),
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),
                        thickness=2)
            cv2.putText(image, "Tilt: " + "{:7.2f}".format(angles[2, 0]),
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),
                        thickness=2)
        if plot_arg == "show":
            matplotlib.pyplot.imshow(image, cmap="gray")
            matplotlib.pyplot.show()
        elif isinstance(plot_arg, matplotlib.axes._subplots.Axes):
            plot_arg.imshow(image, cmap="gray")
        elif isinstance(plot_arg, str):
            matplotlib.pyplot.imshow(image, cmap="gray")
            path = _location.parent/Path("log/"+plot_arg)
            matplotlib.pyplot.savefig(path, dip=1200, format="pdf")
        else:
            raise ValueError(
                "plot_arg must be either 'show' (show the image), an "
                "instance of matplotlib.axes (plot to that axes) or "
                "a string (save to log folder as pdf with that name)")


def calibrate_camera(targets=None, n_reps=1, cams="all"):
    """
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
    """
    # azimuth and elevation of a set of points in camera and world coordinates
    # one list for each camera
    coords = pd.DataFrame(columns=["ele_cam", "azi_cam",
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
    """
    Find linear regression for camera and world coordinates and store
    them in global variables
    """
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
