import sys
sys.path.append("C:/Projects/freefield_toolbox/")
import os
import numpy #for some reason numpy must be imported before PySpin
import PySpin
import cv2
import numpy as np
import dlib
from imutils import face_utils
from pathlib import Path
from freefield_toolbox import setup
import tempfile
import shutil
import glob
import multiprocessing

#define internal variables
_location = Path(__file__).resolve().parents[0]
_dirpath = tempfile.mkdtemp()# create temporary folder to save images
_detector = dlib.get_frontal_face_detector()
_predictor = dlib.shape_predictor(str(_location/Path("shape_predictor_68_face_landmarks.dat")))
_mtx=None #_camera matrix
_dist=None #distortion coefficients
_cam = None
_nodemap = None
_system = None
_pool = None

def init(multiprocess=True):
    global _cam, _nodemap, _system, _pool

    _system = PySpin.System.GetInstance()     # Retrieve singleton reference to system object
    cam_list = _system.GetCameras()    # Retrieve list of _cameras from the system
    #initializing the _camera:
    version = _system.GetLibraryVersion() # Get current library version
    ('PySpin Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))
    num__cameras = cam_list.GetSize()

    if num__cameras != 1:    # Finish if there are no _cameras
        cam_list.Clear()# Clear __camera list before releasing system
        _system.ReleaseInstance() # Release system instance
        raise ValueError('There must be exactly one camera attached to the system!')
    else:
        _cam = cam_list[0]
        nodemap_tldevice = _cam.GetTLDeviceNodeMap()
        node_device_information = PySpin.CCategoryPtr(nodemap_tldevice.GetNode('DeviceInformation'))
        if PySpin.IsAvailable(node_device_information) and PySpin.IsReadable(node_device_information):
            features = node_device_information.GetFeatures()
            for feature in features:
                node_feature = PySpin.CValuePtr(feature)
                setup.printv('%s: %s' % (node_feature.GetName(),
                node_feature.ToString() if PySpin.IsReadable(node_feature) else 'Node not readable'))
        else:
            raise ValueError('Device control information not available.')
        _cam.Init() # Initialize _camera
        _nodemap = _cam.GetNodeMap() # Retrieve GenI_cam nodemap

    if multiprocess:
        n_cores = multiprocessing.cpu_count()
        _pool = multiprocessing.Pool(processes=n_cores)
        setup.printv("using multiprocessing with %s cores" %(n_cores))

def load_calibration(folder):
    global _mtx, _dist
    try:
        _mtx = np.load(Path(folder)/Path("_camera_matrix.npy"))
    except:
        raise FileNotFoundError("could not find _camera matrix")
    try:
        _dist = np.load(Path(folder)/Path("distortion_coefficients.npy"))
    except :
        FileNotFoundError("could not find distortion_coefficients")

def _acquire_images(filename="image", n=1):
    """
    Acquire n images and save them in the temporary folder. Images are saved as
    filename + number of iteration +".jpg"
    """
    node_acquisition_mode = PySpin.CEnumerationPtr(_nodemap.GetNode('AcquisitionMode'))
    if not PySpin.IsAvailable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
        raise ValueError('Unable to set acquisition mode to continuous (enum retrieval). Aborting...')
        return False
    node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
    if not PySpin.IsAvailable(node_acquisition_mode_continuous) or not PySpin.IsReadable(node_acquisition_mode_continuous):
        raise ValueError('Unable to set acquisition mode to continuous (entry retrieval). Aborting...')
        return False
    acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()
    node_acquisition_mode.SetIntValue(acquisition_mode_continuous)
    _cam.BeginAcquisition()
    image_paths =[]
    for i in range(n):
        path = _dirpath+"\\"+filename+str(i)+".jpg"
        image_paths.append(path)
        image_result = _cam.GetNextImage()
        if image_result.IsIncomplete():
            raise ValueError('Image incomplete with image status %d ...' % image_result.GetImageStatus())
        image_converted = image_result.Convert(PySpin.PixelFormat_Mono8, PySpin.HQ_LINEAR)
        image_result.Release()
        image_converted.Save(path)
        setup.printv('Image saved at %s' % path)
    _cam.EndAcquisition()

    return image_paths

def deinit():
    global _cam
    shutil.rmtree(_dirpath) # delete temporary directory
    _cam.DeInit()
    del _cam
    _system.ReleaseInstance()
    setup.printv("Deinitializing _camera...")

def compute_coefficients(images, board_shape=(9,6)):

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    # These are the "coordinates" of the corners of the chess field.
    objp = np.zeros((board_shape[0]*board_shape[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:board_shape[0],0:board_shape[1]].T.reshape(-1,2)
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


def get_headpose(n=4):

    image_paths = _acquire_images(n=n)
    if _pool:
        pose = _pool.map(_get_pose_from_image, image_paths)
    else:
        pose=[]
        for im in image_paths:
            x,y,z = _get_pose_from_image(im)
            pose.append((x,y,z))

    pose = np.array(pose, dtype=np.float)
    pose = np.nanmean(pose, axis=0)
    return pose #x,y,z


def _get_pose_from_image(image_path, plot=False, undistort=False):
    global _mtx, _dist
    im = cv2.imread(image_path)
    if not np.sum(_mtx):
        setup.printv("No _camera matrix loaded!\n"
        "The matrix will be approximated but this is less precise...")
        size = im.shape
        focal_length = size[1]
        center = (size[1]/2, size[0]/2)
        _mtx = np.array(
                             [[focal_length, 0, center[0]],
                             [0, focal_length, center[1]],
                             [0, 0, 1]], dtype = "double"
                             )
    if not np.sum(_dist):
        setup.printv("No distortion coefficients loaded!\n"
        "distortion effects will be ignored...")
        _dist = np.zeros((4,1)) # assuming no lens distortion
        if undistort:
            raise ValueError("ERROR! undistorting images won't work without _camera calibration!")

    if undistort:
        im = undistort_image(im)

    # Array of object points in the world coordinate space.
    # The elements represent points in a generic 3D model
    object_pts = np.float32([[6.825897, 6.760612, 4.402142],
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

    # Get corresponding 2D points in the image:
    face_rects = _detector(im, 0)
    if not face_rects:
        return None, None, None
        raise ValueError("Could not recognize a face in the image %s, returning none") %(image_path)
    shape = _predictor(im, face_rects[0])
    shape = face_utils.shape_to_np(shape)
    image_pts = np.float32([shape[17], shape[21], shape[22],shape[26], shape[36],shape[39],
    shape[42], shape[45], shape[31], shape[35], shape[48], shape[54], shape[57], shape[8]])
    #estimate the translation and rotation coefficients:
    success, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, _mtx, _dist)
    # get the angles out of the transformation matrix:
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    if plot:
        project_points_on_image(im, shape, euler_angle, rotation_vec, translation_vec)
    return euler_angle[0, 0],euler_angle[1, 0],euler_angle[2, 0] # x,y,z

def project_points_on_image(im, shape, euler_angle, rotation_vec, translation_vec):

    reprojectsrc = np.float32([[10.0, 10.0, 10.0],
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
        cv2.circle(im, (x, y), 1, (0, 0, 255), -1)
        cv2.putText(im, "X: " + "{:7.2f}".format(euler_angle[0, 0]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
            0.75, (0, 0, 0), thickness=2)
        cv2.putText(im, "Y: " + "{:7.2f}".format(euler_angle[1, 0]), (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
            0.75, (0, 0, 0), thickness=2)
        cv2.putText(im, "Z: " + "{:7.2f}".format(euler_angle[2, 0]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
            0.75, (0, 0, 0), thickness=2)
        cv2.imshow("Head Pose", im)
    cv2.waitKey(0)

def undistort_image(im):
    h,  w = im.shape[:2]
    new_mtx, roi=cv2.getOptimalNew_cameraMatrix(_mtx,_dist,(w,h),1,(w,h))
    im = cv2.undistort(im, _mtx, _dist, None, new_mtx)
    x,y,w,h = roi
    im = im[y:y+h, x:x+w]  # crop the image
    return im
