import os
import PySpin

system = PySpin.System.GetInstance()     # Retrieve singleton reference to system object
version = system.GetLibraryVersion() # Get current library version
print('PySpin Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))
cam_list = system.GetCameras()    # Retrieve list of cameras from the system
num_cameras = cam_list.GetSize()
if num_cameras != 1:    # Finish if there are no cameras
    cam_list.Clear()# Clear camera list before releasing system
    system.ReleaseInstance() # Release system instance
    print('<Error! There must be exactly one camera attached to the system!')
else:
    cam = cam_list[0]
    nodemap_tldevice = cam.GetTLDeviceNodeMap()
    node_device_information = PySpin.CCategoryPtr(nodemap_tldevice.GetNode('DeviceInformation'))
    if PySpin.IsAvailable(node_device_information) and PySpin.IsReadable(node_device_information):
        features = node_device_information.GetFeatures()
        for feature in features:
            node_feature = PySpin.CValuePtr(feature)
            print('%s: %s' % (node_feature.GetName(),
            node_feature.ToString() if PySpin.IsReadable(node_feature) else 'Node not readable'))
        else:
            print('Device control information not available.')

    cam.Init() # Initialize camera
    nodemap = cam.GetNodeMap() # Retrieve GenICam nodemap

def acquire_images(filename, n=1):
    # In order to access the node entries, they have to be casted to a pointer type (CEnumerationPtr here)
    node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
    if not PySpin.IsAvailable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
        print('Unable to set acquisition mode to continuous (enum retrieval). Aborting...')
        return False
    # Retrieve entry node from enumeration node
    node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
    if not PySpin.IsAvailable(node_acquisition_mode_continuous) or not PySpin.IsReadable(node_acquisition_mode_continuous):
        print('Unable to set acquisition mode to continuous (entry retrieval). Aborting...')
        return False
    # Retrieve integer value from entry node
    acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()
    # Set integer value from entry node as new value of enumeration node
    node_acquisition_mode.SetIntValue(acquisition_mode_continuous)
    # What happens when the camera begins acquiring images depends on the
    # acquisition mode. Single frame captures only a single image, multi
    # frame catures a set number of images, and continuous captures a
    # continuous stream of images
    cam.BeginAcquisition()
    for i in range(n):
        image_result = cam.GetNextImage()
        if image_result.IsIncomplete():
            print('Image incomplete with image status %d ...' % image_result.GetImageStatus())
        image_converted = image_result.Convert(PySpin.PixelFormat_Mono8, PySpin.HQ_LINEAR)
        image_result.Release()
        image_converted.Save(filename)
        print('Image saved at %s' % filename)
        cam.EndAcquisition()

def deinit():
    global cam
    cam.DeInit()        # Deinitialize camera
    print("Deinitializing camera...")

def print_device_info(nodemap):
    """
    This function prints the device information of the camera from the transport
    layer; please see NodeMapInfo example for more in-depth comments on printing
    device information from the nodemap.

    :param nodemap: Transport layer device nodemap.
    :type nodemap: INodeMap
    :returns: True if successful, False otherwise.
    :rtype: bool
    """
    print('*** DEVICE INFORMATION ***\n')
    try:
        result = True
        node_device_information = PySpin.CCategoryPtr(nodemap.GetNode('DeviceInformation'))
        if PySpin.IsAvailable(node_device_information) and PySpin.IsReadable(node_device_information):
            features = node_device_information.GetFeatures()
            for feature in features:
                node_feature = PySpin.CValuePtr(feature)
                print('%s: %s' % (node_feature.GetName(),
                                  node_feature.ToString() if PySpin.IsReadable(node_feature) else 'Node not readable'))
        else:
            print('Device control information not available.')

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False

    return result

if __name__ == "__main__":
    import camera
    camera.init()
    filename="C:/Projects/Spinnaker_Python/tmp.jpg"
    camera.acquire_images(filename)
    camera.deinit()
