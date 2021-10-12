import slab
import numpy as np
import time
import tdt
from numpy import linalg as la
from win32com.client import Dispatch
import freefield
import cv2
hrtf = slab.HRTF.kemar()

def run_exp(goal, s_range, n_trials, vid=True):
    proc = Dispatch('RPco.X')
    proc.ConnectRM1('USB', 1)  # connect processor
    proc.ClearCOF()
    proc.LoadCOF('C:/Users/Everyone/Desktop/TDT/rcx/pulse_generator.rcx')  # load circuit
    # generate sequence of sound source coordinates
    sources = [(np.arange(-s_range, s_range + 1, 10), np.arange(0, s_range + 1, 10))] # use positive elevation sources for now
    az_src = slab.Trialsequence(conditions=sources[0][0], n_reps=n_trials)
    ele_src = slab.Trialsequence(conditions=sources[0][1], n_reps=n_trials)    #todo set trial number correctly
    sound = slab.Binaural.pinknoise(samplerate=48828.125000, duration=10.0) # generate stimulus to load into proc buffer
    tmin = 20  # minimal inter stim interval in ms(if pose and sound source match)
    tmax = 700  # max ISI in ms
    isi_params = {'tmin': tmin, 'tmax': tmax, 'trange': tmax-tmin,
                 'max_dst': la.norm(np.array([s_range, s_range])-np.array([-s_range, -s_range]))} #todo check this (isi doesnt scale correctly)
    # start loop over trials
    proceed = 'y'
    for az, ele in zip(az_src, ele_src):  # get sound source coordinates for next trial
        if proceed == 'y':
            target = np.array([az, ele]) # get target sound position from src list for next trial
            [index] = np.intersect1d(hrtf.cone_sources(az), hrtf.elevation_sources(ele))
            stim=hrtf.apply(index, sound)    # create stim with sound source specific hrtf function applied
            proc.WriteTagV('soundL', 0, stim.data[:, 0])  # laod stim to buffer
            proc.WriteTagV('soundR', 0, stim.data[:, 1])
            proceed = play_trial(target, goal, isi_params, proc, vid) # play n trials #todo fix n_trials
    print('end')
    return 0

def play_trial(target, goal, isi_params, proc, vid=True):
    print('starting..')
    proc.Run() #start processor
    cam = freefield.camera.initialize_cameras('FLIR')    # init cam
    isi, _, _, _ = get_pose_to_isi(target, isi_params, cam) # returns isi from la.norm(pose - target)
    proc.SetTagVal('isi_in', isi)   # send initial isi to tdt processor
    proc.SoftTrg(1)     # buffer trigger (read and play stim)
    proc.SoftTrg(3)     # pulse train trigger #todo make better buffer loop

    while True: #  loop over trials
        isi, pose, info, image = get_pose_to_isi(target, isi_params, cam)  # loop over pose comparison, get isi
        proc.SetTagVal('isi', isi) # write ISI in rcx pulsetrain tag

        if matching_pose(pose, target, goal):  # if pose within target window, start counting time until goal time
            timer = time.time() + goal[1]
            while matching_pose(pose, target, goal):
                isi, pose, info, image = get_pose_to_isi(target, isi_params, cam)  # loop over pose comparison, get isi
                proc.SetTagVal('isi_in', isi) # set isi to pulse train tag
                print('ON TARGET for %i sec')
                if time.time() > timer:
                    break
                    
        if vid:  # optional: show video with marker orientation
            image = vid_marker_orientation(image, pose, info)  # draw marker orientation on frame
            cv2.putText(image, 'TARGET: azimuth: %i, elevation: %i' % (target[0], target[1]), (50, 50),
                        cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 0, 0), lineType=1, thickness=1)  # display target sound location
            cv2.imshow("Image", image)  # show the output frame
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):  # if the `q` key was pressed, break
                break
        print('sound source, az: %i  ele: %i \n head pose, az: %f  ele: %f    ISI: %fs, ' % (
        target[0], target[1], pose[0], pose[1], isi / 1000))  # for testing
    proc.SoftTrg(2)
    proc.Halt()
    cam.halt()
    return input('Goal! Conintue? (y/n)')

def get_pose_to_isi(target, isi_params, cam):
    pose, info, image = pose_from_image(cam)  # compare headpose to sound source target
    isi = isi_from_pose(pose, target, isi_params)  #  use euclidean distance of pose from target to calculate ISI

    return isi, pose, info, image

def isi_from_pose(pose, target, isi_params):
    diff = la.norm(pose - target)
    if np.isnan(diff): # if no marker is detected
        isi = isi_params['imax']
    else:
        #isi = isi_params['imin'] + isi_params['Irange'] * (np.log(diff/isi_params['max_dst']+0.05)+3)/3
        isi = isi_params['imin'] + isi_params['Irange'] * (diff / isi_params['max_dst'])   # scale ISI with deviation of pose from sound source

    return isi

def pose_from_image(cam): # get pose
    #pose = cam.get_head_pose(aruco=True, convert=False) # working alternative, but no access to mtx info
    calibration=0
    convert=0 #todo try in freefield
    resolution=1.0
    average_axis=None

    images = cam.acquire_images(1)
    pose = np.zeros([2, images.shape[2], images.shape[3]])
    for i_cam in range(images.shape[3]):
        for i_image in range(images.shape[2]):
            image = images[:, :, i_image, i_cam]  # get image from array
            if resolution < 1.0:
                image = cam.change_image_res(image, resolution)
            azimuth, elevation, info = cam.model.pose_from_image_aruco(image) #grab info file from headpose.py
            pose[:, i_image, i_cam] = azimuth, elevation
    if convert:
        if not calibration:
            raise ValueError("Camera has to be calibrated to convert coordinates")
        for i_cam in range(pose.shape[2]):  # convert all images for each cam ...
            for i_angle, angle in enumerate(["azimuth", "elevation"]):  # ... and each angle
                a, b = cam.calibration[f"cam{i_cam}"][angle]["a"], cam.calibration[f"cam{i_cam}"][angle]["b"]
                pose[i_angle, :, i_cam] = a + b * pose[i_angle, :, i_cam]
    if average_axis is not None:
        pose = pose.mean(axis=average_axis)

    return pose, info, images[:,:,0,0]

def matching_pose(pose, target, goal):    # criteria to end experiment (pose matches sound source) #todo find better way
    if target[0] - goal[0] <= pose[0] <= target[0] + goal[0] and \
            target[1] - goal[0] <= pose[1] <= target[1] + goal[0]:
        return 1

    else: return 0

def vid_marker_orientation(image, pose, info):
    marker_len = .05
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)
    (corners, ids, rejected) = cv2.aruco.detectMarkers(image,arucoDict)
    if len(corners) > 0:
        imaxis = cv2.aruco.drawDetectedMarkers(image.copy(), corners, ids)
        image = cv2.aruco.drawAxis(imaxis, info[0], info[1], info[2], info[3], marker_len)
        # info: list of arrays [camera_matrix, dist_coeffs, rotation_vec, translation_vec]
        bottomLeftCornerOfText = (int(corners[0][0][2][0]), int(corners[0][0][2][1]))
        cv2.putText(image, 'azimuth: %f elevation: %f' % (pose[0], pose[1]), # display heade pose
        bottomLeftCornerOfText, cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(225, 225, 225),
                    lineType=1, thickness=1)

    return(image)

if __name__ == "__main__":
    run_exp([3,3],30,3, vid=True)