# Tests to run on the real setup
# start with the basic elements, then run complete procedures like
# calibration or localization test
from freefield import freefield, cameras
import PySpin
import numpy as np
import time
from matplotlib import pyplot as plt

#### 1. Test if the camera API is working properly ####
# initialize he cams
system = PySpin.System.GetInstance()
cams = system.GetCameras()
ncams = cams.GetSize()
for cam in cams:
    cam.Init()  # Initialize camera
# take an image from each camera:
fig, ax = plt.subplots(ncams)
for i, cam in enumerate(cams):  # start the cameras
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
    time.sleep(0.01)
    image_result = cam.GetNextImage()
    if image_result.IsIncomplete():
        raise ValueError('Image incomplete: image status %d ...'
                         % image_result.GetImageStatus())
    image = image_result.Convert(
        PySpin.PixelFormat_Mono8, PySpin.HQ_LINEAR)
    image = image.GetNDArray()
    image_result.Release()
    ax[i].imshow(image, cmap="gray")
[cam.EndAcquisition() for cam in cams]
plt.show()

# check if taking multiple images and works as expected
freefield.initialize_setup(setup="dome", default_mode="cam_calibration", camera_type="flir")
cams = freefield.CAMERAS
images=[]
for i in range(9):
    freefield.wait_for_button()
    im = cams.acquire_images()
    images.append(im[:,:,0,0])
# check the headpose
for image in images:
    pose = cams.model.pose_from_image(image)
    print(pose)



#### 2. test if reading from and writing to the device works ####
freefield.initialize_setup(setup="dome", default_mode="play_rec", camera_type="flir")
freefield.write(tag="playbuflen", value=100, procs="RX8s")
signal = np.random.randn(100)
freefield.write(tag="data", value=signal, procs="RX8s")
rec = freefield.read(tag="data", n_samples=100, proc="RX81")
assert all(rec.round(3) == signal.round(3))
