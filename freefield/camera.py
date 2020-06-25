import numpy  # for some reason numpy must be imported before PySpin
try:
    import PySpin
except ModuleNotFoundError:
    print("PySpin module required for working with FLIR cams not found! \n"
          "You can download the .whl here: \n"
          "https://www.flir.com/products/spinnaker-sdk/")
import cv2
import dlib
from imutils import face_utils
from pathlib import Path
from freefield import setup
import multiprocessing
from slab.psychoacoustics import Trialsequence
import time
import matplotlib
from scipy import stats
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

# define internal variables
_location = Path(__file__).resolve().parents[0]
_detector = dlib.get_frontal_face_detector()
# check if this file exists, if not download it:
_predictor = dlib.shape_predictor(
    str(_location/Path("shape_predictor_68_face_landmarks.dat")))
# regression coefficients for camera vs target coordinates for azimuth and
# elevation. Tuple with (slope, intercept) for each camera
_ele_reg = []
_azi_reg = []
_cam_type = None
_system = None
_pool = None
_cal = None
_object_pts = numpy.float32([[6.825897, 6.760612, 4.402142],
                             [1.330353, 7.122144, 6.903745],
                             [-1.330353, 7.122144, 6.903745],
                             [-6.825897, 6.760612, 4.402142],
                             [5.311432, 5.485328, 3.987654],
                             [1.789930, 5.393625, 4.413414],
                             [-1.789930, 5.393625, 4.413414],
                             [-5.311432, 5.485328, 3.987654],
                             [2.005628, 1.409845, 6.165652],
                             [-2.005628, 1.409845, 6.165652],
                             [2.774015, -2.080775, 5.048531],
                             [-2.774015, -2.080775, 5.048531],
                             [0.000000, -3.116408, 6.097667],
                             [0.000000, -7.415691, 4.070434]])

_reprojectsrc = numpy.float32([[10.0, 10.0, 10.0],
                               [10.0, 10.0, -10.0],
                               [10.0, -10.0, -10.0],
                               [10.0, -10.0, 10.0],
                               [-10.0, 10.0, 10.0],
                               [-10.0, 10.0, -10.0],
                               [-10.0, -10.0, -10.0],
                               [-10.0, -10.0, 10.0]])


def init(multiprocess=False, type="freefield"):
    global _cams, _system, _pool, _cam_type
    _cam_type = type.lower()
    if _cam_type == "freefield":  # Use FLIR cameras
        _system = PySpin.System.GetInstance()  # get reference to system object
        _cams = _system.GetCameras()   # get list of _cameras from the system
        # initializing the _camera:
        num_cameras = _cams.GetSize()
        if num_cameras == 0:    # Finish if there are no cameras
            _cams.Clear()  # Clear camera list before releasing system
            _system.ReleaseInstance()  # Release system instance
            raise ValueError('No camera found!')
        else:
            for cam in _cams:
                cam.Init()  # Initialize camera
            setup.printv("initialized %s FLIR camera(s)" % (len(_cams)))
    elif _cam_type == "web":
        _cams = []
        stop = False
        i = 0
        while stop is False:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                _cams.append(cap)
            else:
                stop = True
        setup.printv("initialized %s webcams(s)" % (len(_cams)))
    else:
        raise ValueError("type must be either 'freefield' or 'web'")

    if multiprocess:
        n_cores = multiprocessing.cpu_count()
        _pool = multiprocessing.Pool(processes=n_cores)
        setup.printv("using multiprocessing with %s cores" % (n_cores))


def acquire_image(cams="all"):
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
    images = []  # pre allocate memory space to make faster?
    for cam in cams:
        if _cam_type == "freefield":
            node_acquisition_mode = PySpin.CEnumerationPtr(
                cam.GetNodeMap().GetNode('AcquisitionMode'))
            if (not PySpin.IsAvailable(node_acquisition_mode) or
                    not PySpin.IsWritable(node_acquisition_mode)):
                raise ValueError(
                    'Unable to set acquisition to continuous, aborting...')
            node_acquisition_mode_continuous = \
                node_acquisition_mode.GetEntryByName('Continuous')
            if (not PySpin.IsAvailable(node_acquisition_mode_continuous) or
                    not PySpin.IsReadable(
                    node_acquisition_mode_continuous)):
                raise ValueError(
                    'Unable to set acquisition to continuous, aborting...')
            acquisition_mode_continuous = \
                node_acquisition_mode_continuous.GetValue()
            node_acquisition_mode.SetIntValue(acquisition_mode_continuous)
            cam.BeginAcquisition()

            image_result = cam.GetNextImage()
            if image_result.IsIncomplete():
                raise ValueError('Image incomplete: image status %d ...'
                                 % image_result.GetImageStatus())
            image = image_result.Convert(
                PySpin.PixelFormat_Mono8, PySpin.HQ_LINEAR)
            image = image.GetNDArray()
            image_result.Release()
            cam.EndAcquisition()

        elif _cam_type == "web":
            # The webcam takes several pictures and reading only advances
            # the buffer one step at a time, thus grab all images and only
            # retrieve the latest one
            for i in range(cv2.CAP_PROP_FRAME_COUNT):
                cam.grab()
            ret, image = cam.retrieve()
            if ret is False:
                setup.printv("could not acquire image, returning None...")
        images.append(image)
    return images


def get_headpose(cams="all", convert=False, average=False, n=1):
    """
    Acquire n images and compute headpose (elevation and azimuth). If
    convert is True use the regression coefficients to convert
    the camera into world coordinates
    """
    # TODO: sanity check the resulting dataframe, e.g. how big is the max diff
    pose = pd.DataFrame(columns=["ele", "azi", "cam"])
    for n in range(n):
        images = acquire_image(cams)  # take images
        for i, image in enumerate(images):
            ele, azi, _ = _pose_from_image(image)
            row = pd.DataFrame([[ele, azi, i]],
                               columns=["ele", "azi", "cam"])
            pose = pose.append(row)
    if convert:  # convert azimuth and elevation to world coordinates
        if _cal is None:
            raise ValueError("Can't convert coordinates because camera is"
                             "not calibrated!")
        else:
            for cam in np.unique(pose["cam"]):
                for angle in ["azi", "ele"]:
                    reg = _cal[(_cal["cam"] == cam) & (_cal["angle"] == angle)]
                    pose.loc[pose["cam"] == cam, angle] = \
                        (pose[pose["cam"] == cam][angle] -
                         reg["a"].values)/reg["b"].values
        pose.insert(3, "frame", "world")
    else:
        pose.insert(3, "frame", "camera")
    if average:  # only return the mean
        if not convert:
            raise ValueError("Can only average after convertig coordinates!")
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
    # approximate camera matrix:
    focal_length = image.shape[1]
    center = (image.shape[1]/2, image.shape[0]/2)
    mtx = numpy.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double")
    dist = numpy.zeros((4, 1))  # assuming no lens distortion

    # Get corresponding 2D points in the image:
    face_rects = _detector(image, 0)
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
    success, rotation_vec, translation_vec = \
        cv2.solvePnP(_object_pts, image_pts, mtx, dist)
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
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
            cv2.putText(image, "Elevation: " + "{:7.2f}".format(angles[0, 0]),
                        (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0),
                        thickness=2)
            cv2.putText(image, "Azimuth: " + "{:7.2f}".format(angles[1, 0]),
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0),
                        thickness=2)
            cv2.putText(image, "Tilt: " + "{:7.2f}".format(angles[2, 0]),
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0),
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


def calibrate_camera(targets=None, n_reps=1):
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
    coords = pd.DataFrame(columns=["ele", "azi", "cam", "frame", "n"])
    if _cam_type == "web" and targets is None:
        raise ValueError("Define target positions for calibrating webcam!")
    elif _cam_type is None:
        raise ValueError("Initialize Camera before calibration!")
    elif _cam_type == "freefield":
        targets = setup.all_leds()  # get the speakers that have a LED attached
    if not setup._mode == "camera_calibration":  # initialize setup
        setup.initialize_devices(mode="camera_calibration")
    seq = Trialsequence(name="cam", n_reps=n_reps, conditions=targets)
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
        pose = get_headpose(average=False, convert=False)
        pose.insert(4, "n", seq.this_n)
        coords = coords.append(pose, ignore_index=True, sort=True)
        coords = coords.append(
            pd.DataFrame([[ele, azi, "world", seq.this_n]],
                         columns=["ele", "azi", "frame", "n"]),
            ignore_index=True, sort=True)
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
    _cal = pd.DataFrame(columns=["a", "b", "cam", "angle"])
    if plot:
        fig, ax = plt.subplots(2)
        fig.suptitle("World vs Camera Coordinates")
    # find the entries which which contain NaN for elevation and azimuth
    bads = coords[coords["ele"].isna()].index
    for bad in bads:
        row = coords.loc[bad]
        # get the position of the target in world coordinates
        pos = coords[np.logical_and(coords["frame"] == "world",
                                    coords["n"] == row["n"])]
        setup.printv("Dropping NaN entry for cam %s at elevation %s and "
                     "azimuth %s" % (row.cam, pos.ele.values[0],
                                     pos.azi.values[0]))
    coords = coords.drop(bads)  # remove all NaN entires
    # bad_n = coords.n.value_counts()[coords.n.value_counts() == 1].index

    for cam in pd.unique(coords["cam"].dropna()):  # calibrate each camera
        cam_coords = coords[coords["cam"] == cam]
        world_coords = coords[coords['n'].isin(cam_coords['n']) &
                              (coords["frame"] == "world")]
        # find regression coefficients for azimuth and elevation
        for i, angle in enumerate(["ele", "azi"]):
            y = cam_coords[angle].values.astype(float)
            x = world_coords[angle].values.astype(float)
            b, a, r, _, _ = stats.linregress(x, y)
            if np.abs(r) < 0.85:
                setup.printv("For cam %s %s correlation between camera and"
                             "world coordinates is only %s! \n"
                             "There might be something wrong..."
                             % (cam, angle, r))
            _cal = \
                _cal.append(pd.DataFrame([[a, b, cam, angle]],
                                         columns=["a", "b", "cam", "angle"]),
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
