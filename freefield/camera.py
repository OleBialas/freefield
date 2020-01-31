import numpy #for some reason numpy must be imported before PySpin
import PySpin
import cv2
import dlib
from imutils import face_utils
from pathlib import Path
from freefield import setup
import multiprocessing
import slab

#define internal variables
_location = Path(__file__).resolve().parents[0]
_detector = dlib.get_frontal_face_detector()
# check if this file exists, if not download it:
_predictor = dlib.shape_predictor(str(_location/Path("shape_predictor_68_face_landmarks.dat")))
_mtx=None #_camera matrix
_dist=None #distortion coefficients
_cam_list = None
_system = None
_pool = None
_calibration=None
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

def init(multiprocess=False):
    global _cam_list, _system, _pool
    _system = PySpin.System.GetInstance()     # Retrieve singleton reference to system object
    _cam_list = _system.GetCameras()    # Retrieve list of _cameras from the system
    #initializing the _camera:
    version = _system.GetLibraryVersion() # Get current library version
    setup.printv('PySpin Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))
    num_cameras = _cam_list.GetSize()
    if num_cameras == 0:    # Finish if there are no cameras
        _cam_list.Clear() # Clear camera list before releasing system
        _system.ReleaseInstance() # Release system instance
        raise ValueError('No camera found!')
    else:
        for cam in _cam_list:
            nodemap_tldevice = cam.GetTLDeviceNodeMap()
            node_device_information = PySpin.CCategoryPtr(nodemap_tldevice.GetNode('DeviceInformation'))
            if PySpin.IsAvailable(node_device_information) and PySpin.IsReadable(node_device_information):
                features = node_device_information.GetFeatures()
                for feature in features:
                    node_feature = PySpin.CValuePtr(feature)
                    setup.printv('%s: %s' % (node_feature.GetName(),
                    node_feature.ToString() if PySpin.IsReadable(node_feature) else 'Node not readable'))
            else:
                raise ValueError('Device control information not available.')
            cam.Init() # Initialize camera

    if multiprocess:
        n_cores = multiprocessing.cpu_count()
        _pool = multiprocessing.Pool(processes=n_cores)
        setup.printv("using multiprocessing with %s cores" %(n_cores))

def _acquire_images(n=1):
    """
    Acquire n images from each active camera. The images are returned as arrays in list of list of shape m x n where
    m is the number of cameras and n is the number of trials
    """
    global _cam_list
    if _cam_list is None:
        raise ValueError("Cameras must be initialized before acquisition")
    else:
        all_images=[]
        for cam in _cam_list:
            node_acquisition_mode = PySpin.CEnumerationPtr(cam.GetNodeMap().GetNode('AcquisitionMode'))
            if not PySpin.IsAvailable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
                raise ValueError('Unable to set acquisition mode to continuous (enum retrieval). Aborting...')
            node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
            if not PySpin.IsAvailable(node_acquisition_mode_continuous) or not PySpin.IsReadable(node_acquisition_mode_continuous):
                raise ValueError('Unable to set acquisition mode to continuous (entry retrieval). Aborting...')
            acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()
            node_acquisition_mode.SetIntValue(acquisition_mode_continuous)
            cam.BeginAcquisition()
            images =[]
            for i in range(n):
                image_result = cam.GetNextImage()
                if image_result.IsIncomplete():
                    raise ValueError('Image incomplete with image status %d ...' % image_result.GetImageStatus())
                image_converted = image_result.Convert(PySpin.PixelFormat_Mono8, PySpin.HQ_LINEAR)
                image_result.Release()
                images.append(image_converted.GetNDArray())
            cam.EndAcquisition()
            all_images.append(images)
        return all_images

def deinit():
    global _cam_list
    for cam in _cam_list:
        cam.DeInit()
        del cam
    _cam_list.Clear()
    _system.ReleaseInstance()
    setup.printv("Deinitializing _camera...")

def get_headpose(n=4):
    """
    Take n images, estimate the headpose from each image and return the
    average result. returns 3xn array where n is the number of cameras attached
    to the system.
    """

    images = _acquire_images(n) # take images
    headpose = []
    for cam_nr in range(_cam_list.GetSize()):
        tmp=numpy.zeros([n,3])
        for i in range(n):
            tmp[i] = _get_pose_from_image(images[cam_nr][i])
        headpose.append(tmp)
    headpose = _average_headpose(headpose) # get the mean

    return headpose

def _average_headpose(headpose):
    if len(headpose)==1:
        return headpose.mean(axis=0)
    if len(headpose) > 1:
        raise ValueError("multiple cameras currently not implemented!")

def _get_pose_from_image(image, plot=False):
    global _mtx, _dist
    if not numpy.sum(_mtx):
        setup.printv("No _camera matrix loaded!\n"
        "The matrix will be approximated but this is less precise...")
        size = image.shape
        focal_length = size[1]
        center = (size[1]/2, size[0]/2)
        _mtx = numpy.array(
                             [[focal_length, 0, center[0]],
                             [0, focal_length, center[1]],
                             [0, 0, 1]], dtype = "double"
                             )
    if not numpy.sum(_dist):
        setup.printv("No distortion coefficients loaded!\n"
        "distortion effects will be ignored...")
        _dist = numpy.zeros((4,1)) # assuming no lens distortion

    # Get corresponding 2D points in the image:
    face_rects = _detector(image, 0)
    if not face_rects:
        printv("Could not recognize a face in the image, returning none")
        return None, None, None
    shape = _predictor(image, face_rects[0])
    shape = face_utils.shape_to_np(shape)
    image_pts = numpy.float32([shape[17], shape[21], shape[22],shape[26], shape[36],shape[39],
    shape[42], shape[45], shape[31], shape[35], shape[48], shape[54], shape[57], shape[8]])
    #estimate the translation and rotation coefficients:
    success, rotation_vec, translation_vec = cv2.solvePnP(_object_pts, image_pts, _mtx, _dist)
    # get the angles out of the transformation matrix:
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)
    if plot:
        project_points_on_image(image, shape, euler_angle, rotation_vec, translation_vec)
    return euler_angle[0, 0],euler_angle[1, 0],euler_angle[2, 0] # x,y,z

def project_points_on_image(image, shape, euler_angle, rotation_vec, translation_vec):

    reprojectsrc = numpy.float32([[10.0, 10.0, 10.0],
                            [10.0, 10.0, -10.0],
                            [10.0, -10.0, -10.0],
                            [10.0, -10.0, 10.0],
                            [-10.0, 10.0, 10.0],
                            [-10.0, 10.0, -10.0],
                            [-10.0, -10.0, -10.0],
                            [-10.0, -10.0, 10.0]])

    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, _mtx, _dist)
    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

    for (x, y) in shape:
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
        cv2.putText(image, "X: " + "{:7.2f}".format(euler_angle[0, 0]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
            0.75, (255, 255, 255), thickness=2)
        cv2.putText(image, "Y: " + "{:7.2f}".format(euler_angle[1, 0]), (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
            0.75, (255, 255, 255), thickness=2)
        cv2.putText(image, "Z: " + "{:7.2f}".format(euler_angle[2, 0]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
            0.75, (255, 255, 255), thickness=2)
        cv2.imshow("Head Pose", image)
    cv2.waitKey(0)

def calibrate_camera(positions, speaker_config="dome"):
    """"
    Makes LEDs light up at the given postions. Subject has to align their head
    with the lit LED and push the button so a picture is taken and the head pose is
    determined. Then we can determine the coefficients of the linear regression for
    led position vs measured head position
    """
    from sklearn.linear_model import LinearRegression
    from matplotlib import pyplot as plt
    rp2_file = _location.parents[0] / Path("examples/button_response.rcx")
    rx8_file = _location.parents[0] / Path("examples/to_bits.rcx")
    setup.set_speaker_config(speaker_config)
    setup.initialize_devices(RX81_file=rx8_file, RP2_file=rp2_file, ZBus=True, cam=True)
    leds = setup.all_leds()
    seq = slab.psychoacoustics.Trialsequence(name="camera calibration", n_reps=5, conditions=range(len(leds)))
    while not seq.finished:
        row = leds[seq.__next__()]
        setup.printv("trial nr %s: speaker at azimuth of %s and elevation of %s"%(seq.this_n,row[3],row[4]))
        setup.set_variable(variable="set_zero", value=False, proc="RX81")
        set_variable(variable="bitval", value=int(i), proc="RX81")
        tic = time.time()
        while not get_variable(variable="response", proc="RP2"):
            time.sleep(0.1)  # wait untill button is pressed
        set_variable(variable="set_zero", value=True, proc="RX81")
        x, y, z = camera.get_headpose(n=5)
        if y is not None:
            results[count, 1] = y
            results[count, 0] = pos[numpy.where(bits == i)[0][0]]

    linear_regressor = LinearRegression()
    linear_regressor.fit(results[:, 0].reshape(-1, 1), results[:, 1].reshape(-1, 1))
    pred = linear_regressor.predict(results[:, 0].reshape(-1, 1))

    if plot:
        plt.scatter(results[:, 0].reshape(-1, 1), results[:, 1].reshape(-1, 1))
        plt.plot(results[:, 0].reshape(-1, 1), pred)
        plt.show()

    return results, linear_regressor


"""
def calibrate_headpose(n_repeat=5, bits=numpy.array([8, 4, 2, 1]), pos=numpy.array([64.20, 47.08, 25.68, 0]),
                       plot=True):
	Makes LEDs light up at the given postions. Subject has to align their head
	with the lit LED and push the button so a picture is taken and the head pose is
	determined. Then we can determine the coefficients of the linear regression for
	led position vs measured head position
    from sklearn.linear_model import LinearRegression
    from matplotlib import pyplot as plt
    rp2_file = _location.parents[0] / Path("examples/button_response.rcx")
    rx8_file = _location.parents[0] / Path("examples/to_bits.rcx")
    if not _speaker_config:
        set_speaker_config("arc")
    initialize_devices(RX81_file=rx8_file, RP2_file=rp2_file, ZBus=True, cam=True)
    trials = numpy.tile(bits, n_repeat)
    results = numpy.zeros([len(trials), 2])
    results[:, 0] = trials
    for i, count in zip(trials, range(len(trials))):
        print("trial nr" + str(count))
        set_variable(variable="set_zero", value=False, proc="RX81")
        set_variable(variable="bitval", value=int(i), proc="RX81")
        tic = time.time()
        while not get_variable(variable="response", proc="RP2"):
            time.sleep(0.1)  # wait untill button is pressed
        set_variable(variable="set_zero", value=True, proc="RX81")
        x, y, z = camera.get_headpose(n=5)
        if y is not None:
            results[count, 1] = y
            results[count, 0] = pos[numpy.where(bits == i)[0][0]]

    linear_regressor = LinearRegression()
    linear_regressor.fit(results[:, 0].reshape(-1, 1), results[:, 1].reshape(-1, 1))
    pred = linear_regressor.predict(results[:, 0].reshape(-1, 1))

    if plot:
        plt.scatter(results[:, 0].reshape(-1, 1), results[:, 1].reshape(-1, 1))
        plt.plot(results[:, 0].reshape(-1, 1), pred)
        plt.show()

    return results, linear_regressor



#Not working yet --> Calculate the camera coefficients from chessboard
#images and use them to undistort he recorded images:
def compute_coefficients(images, board_shape=(9,6)):

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    # These are the "coordinates" of the corners of the chess field.
    objp = numpy.zeros((board_shape[0]*board_shape[1],3), numpy.float32)
    objp[:,:2] = numpy.mgrid[0:board_shape[0],0:board_shape[1]].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            #get the position of the corners in the
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (9,6), corners2,ret)
            cv2.imshow('img',img)
            cv2.waitKey(500)
        else:
            setup.printv("could not find a chess board in the image "+fname)

    cv2.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrate_camera(objpoints, imgpoints, gray.shape[::-1],None,None)
    return mtx, dist

def load_calibration(folder):
    global _mtx, _dist
    try:
        _mtx = numpy.load(Path(folder)/Path("_camera_matrix.npy"))
    except:
        raise FileNotFoundError("could not find _camera matrix")
    try:
        _dist = numpy.load(Path(folder)/Path("distortion_coefficients.npy"))
    except :
        FileNotFoundError("could not find distortion_coefficients")

def undistort_image(im):
    h,  w = im.shape[:2]
    new_mtx, roi=cv2.getOptimalNew_cameraMatrix(_mtx,_dist,(w,h),1,(w,h))
    im = cv2.undistort(im, _mtx, _dist, None, new_mtx)
    x,y,w,h = roi
    im = im[y:y+h, x:x+w]  # crop the image
    return im
"""