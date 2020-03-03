import numpy  # for some reason numpy must be imported before PySpin
import PySpin
import cv2
import dlib
from imutils import face_utils
from pathlib import Path
from freefield import setup
import multiprocessing
import slab
import time
import matplotlib
import scipy
from matplotlib import pyplot as plt

# define internal variables
_location = Path(__file__).resolve().parents[0]
_detector = dlib.get_frontal_face_detector()
# check if this file exists, if not download it:
_predictor = dlib.shape_predictor(
    str(_location/Path("shape_predictor_68_face_landmarks.dat")))
_mtx = None  # _camera matrix
_dist = None  # distortion coefficients
_cams = None
_ele_reg = None  # regression coefficients for camera vs target coordinates ...
_azi_reg = None  # ...for elevation and azimuth respectively
_cam_type = None
_system = None
_pool = None
_calibration = None
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


def acquire_image(cam_idx=0):
    """
    acquire an image from a camera. The camera at the index given by cam
    idx in the camera list _cams is used. If only one camera is being used
    the default cam_idx=0 is sufficient.
    """
    cam = _cams[cam_idx]
    if _cam_type is None:
        raise ValueError("Cameras must be initialized before acquisition")
    elif _cam_type == "freefield":
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
    return image


def get_headpose(cam_idx=0, convert_coordinates=False, n_average=1):
    """
    Acquire n images and compute headpose (elevation and azimuth). If
    convert_coordinates is True use the regression coefficients to convert
    the camera into world coordinates
    """
    elevation, azimuth = 0, 0
    for i in range(n_average):
        image = acquire_image(cam_idx)  # take images
        ele, azi, _ = pose_from_image(image)
        elevation += ele
        azimuth += azi
    elevation /= n_average
    azimuth /= n_average
    if convert_coordinates:  # y= b * x +c
        elevation = _ele_reg[0] * elevation + _ele_reg[0]
        azimuth = _azi_reg[0] * azimuth + _azi_reg[0]
    return azimuth, elevation


def plot_pose(image, euler_angle, shape, rotation_vec, translation_vec,
              _mtx, _dist, plot_arg="show"):
    """
    Acquire an image, compute the headpose and then plot the acquired image
    with the fitted mask of model points and the computed angles
    """

    reprojectdst, _ = cv2.projectPoints(_reprojectsrc, rotation_vec,
                                        translation_vec, _mtx, _dist)
    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

    for (x, y) in shape:
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
        cv2.putText(image, "Elevation: " + "{:7.2f}".format(euler_angle[0, 0]),
                    (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0),
                    thickness=2)
        cv2.putText(image, "Azimuth: " + "{:7.2f}".format(euler_angle[1, 0]),
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0),
                    thickness=2)
        cv2.putText(image, "Tilt: " + "{:7.2f}".format(euler_angle[2, 0]),
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
        raise ValueError("plot_arg must be either 'show' (show the image), an "
                         "instance of matplotlib.axes (plot to that axes) or "
                         "a string (save to log folder as pdf with that name)")


def pose_from_image(image, only_euler=True):
    """
    Compute the head pose from an image, which must be a 2d (grayscale) or
    3d (color) numpy array. If only_euler=True (default), only angles azimuth,
    elevation and tilt are returned. If False also face shape, rotation vector,
    translation vector, camera matrix , and distortion are returned. This is
    nesseccary to plot the image with the computed headpose.
    """
    global _mtx, _dist
    if not numpy.sum(_mtx):
        size = image.shape
        focal_length = size[1]
        center = (size[1]/2, size[0]/2)
        _mtx = numpy.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )
    if not numpy.sum(_dist):
        _dist = numpy.zeros((4, 1))  # assuming no lens distortion

    # Get corresponding 2D points in the image:
    face_rects = _detector(image, 0)
    if not face_rects:
        setup.printv("Could not recognize a face in the image, returning none")
        if only_euler:
            return None, None, None
        else:
            return None, None, None, None, None, None
    shape = _predictor(image, face_rects[0])
    shape = face_utils.shape_to_np(shape)
    image_pts = numpy.float32([shape[17], shape[21], shape[22], shape[26],
                               shape[36], shape[39], shape[42], shape[45],
                               shape[31], shape[35], shape[48], shape[54],
                               shape[57], shape[8]])
    # estimate the translation and rotation coefficients:
    success, rotation_vec, translation_vec = \
        cv2.solvePnP(_object_pts, image_pts, _mtx, _dist)
    # get the angles out of the transformation matrix:
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)
    if only_euler:
        # elevation, azimuth and tilt
        return euler_angle[0, 0], euler_angle[1, 0], euler_angle[2, 0]
    else:
        return euler_angle, shape, rotation_vec, translation_vec, _mtx, _dist


def calibrate_webcam(targets, n_repeat=1):
    """
    Calibration of the camera for webcams: find the coefficients for the
    regression between the camera and world coordinates. You will be asked
    to point your head towards the target positions in randomized order and
    press enter.
    Attributes:
    target_positions (list of tuples): elevation and azimuth for any number of
        points in world coordinates
    n_repeat (int): number of repetitions for each target (default = 1)
    """
    targets_camera = []
    targets_world = []
    if not _cam_type == "web":
        raise ValueError("This functions is for calibrating webcams only!")
    seq = slab.psychoacoustics.Trialsequence(
        name="camera calibration", n_reps=n_repeat,
        conditions=range(len(targets)))
    while seq.n_remaining:
        pos = targets[seq.__next__()]
        input("point your head towards the target at elevation: %s and "
              "azimuth %s. \n Then press enter to take an image an get "
              "the headpose" % (pos[0], pos[1]))
        image = acquire_image(0)
        ele, azi, _ = pose_from_image(image)
        if ele is not None and azi is not None:
            targets_camera.append((ele, azi))
            targets_world.append(pos)
    camera_to_world(targets_world, targets_camera)
    return targets_world, targets_camera


def camera_to_world(world_coordinates, camera_coordinates, plot=True):

    global _ele_reg, _azi_reg
    if plot:
        fig, ax = plt.subplots(2)
        fig.suptitle("World vs Camera Coordinates")
    for i, angle in enumerate(["elevation", "azimuth"]):
        x = numpy.array([w[i] for w in world_coordinates])
        y = numpy.array([c[i] for c in camera_coordinates])
        slope, intercept, r, _, _ = scipy.stats.linregress(x, y)
        if angle == "elevation":
            _ele_reg = (slope, intercept)
        elif angle == "azimuth":
            _azi_reg = (slope, intercept)
        if plot:
            ax[i].scatter(x, y, c="black")
            ax[i].plot(x, x*slope+intercept, c="black", linestyle="--")
            ax[i].set_title(angle)
            ax[i].set_xlabel("world coordinates in degree")
            ax[i].set_ylabel("camera coordinates in degree")
    if plot:
        plt.show()


def calibrate_freefieldcam(targets, repetitions):
    """"
    Makes LEDs light up at the given postions. Subject has to align their head
    with the lit LED and push the button so a picture is taken and the head
    pose is determined. Then we can determine the coefficients of the linear
    regression for led position vs measured head position
    """
    rp2_file = _location.parents[0] / Path("rcx/button_response.rcx")
    rx8_file = _location.parents[0] / Path("rcx/leds.rcx")
    setup.initialize_devices(RX8_file=rx8_file, RP2_file=rp2_file,
                             ZBus=True, cam=True)
    leds = setup.all_leds()
    seq = slab.psychoacoustics.Trialsequence(
        name="camera calibration", n_reps=4, conditions=range(len(leds)))
    results = \
        [numpy.zeros([seq.n_trials, 4]) for i in range(_cams.GetSize())]
    while not seq.finished:
        row = leds[seq.__next__()]
        setup.printv("trial nr %s: speaker at azi: %s and ele: of %s" %
                     (seq.this_n, row[3], row[4]))
        setup.set_variable(variable="bitmask", value=row[5], proc=int(row[2]))
        while not setup.get_variable(variable="response", proc="RP2"):
            time.sleep(0.1)  # wait untill button is pressed
        headpose, std = get_headpose(n=4)
        setup.set_variable(variable="bitmask", value=0, proc=int(row[2]))
        for i, h in enumerate(headpose):
            results[i][seq.this_n][0:2] = row[3:5]
            results[i][seq.this_n][2:] = h[0:2].round(2)  # estimated x and y

    # for i in range(len(results)):
        results[i] = results[i][~numpy.isnan(results[i]).any(axis=1)]

    # Now fit regression to data and plot results:
    fig, ax = plt.subplots(_cams.GetSize(), sharex=True, sharey=True)
    fig.suptitle("Horizontal")
    for i, r in enumerate(results):
        r = r[~numpy.isnan(r).any(axis=1)]
        linear_regressor.fit(r[:, 0].reshape(-1, 1), r[:, 2].reshape(-1, 1))
        pred = linear_regressor.predict(r[:, 0].reshape(-1, 1))
        ax[i].scatter(r[:, 0], r[:, 2])
        ax[i].plot(r[:, 0], pred[:, 0])

    fig, ax = plt.subplots(_cams.GetSize(), sharex=True, sharey=True)
    fig.suptitle("Vertical")
    for i, r in enumerate(results):
        r = r[~numpy.isnan(r).any(axis=1)]
        linear_regressor.fit(r[:, 1].reshape(-1, 1), r[:, 3].reshape(-1, 1))
        pred = linear_regressor.predict(r[:, 1].reshape(-1, 1))
        ax[i].scatter(r[:, 1], r[:, 3])
        ax[i].plot(r[:, 1], pred[:, 0])

    return results, linear_regressor


def deinit():
    global _cams
    if _cam_type == "freefield":
        for cam in _cams:
            cam.EndAcquisition()
            cam.DeInit()
            del cam
        _cams.Clear()
        _system.ReleaseInstance()
    if _cam_type == "web":
        for cam in _cams:
            cam.release()
    setup.printv("Deinitializing _camera...")
