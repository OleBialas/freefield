import slab
import numpy as np
import time
import tdt
from numpy import linalg as la
from win32com.client import Dispatch
import freefield
import cv2
hrtf = slab.HRTF.kemar()

#import win32com.client
#import imutils
#import headpose

# goal = [6,0.5]  # end conditions: target window size (az/ele angles) + timer
# s_range = 30   # set range of horizontal and vertical speaker locations (eg 30: -30° to 30° in elevation and azimuth)
# n_trials = 1
# run_exp([6,0.5],30,1)

def run_exp(goal, s_range, n_trials, vid=True):
    proc = Dispatch('RPco.X')
    proc.ConnectRM1('USB', 1)  # connect processor
    proc.ClearCOF()
    proc.LoadCOF('C:/Users/Everyone/Desktop/TDT/rcx/pulse_generator.rcx')  # load circuit

    # generate sequence of sound source coordinates
    #sources = np.stack((np.arange(-s_range, s_range + 1, 10), np.arange(-s_range, s_range + 1, 10)))
    sources = np.stack((np.arange(0, s_range + 1, 10), np.arange(0, s_range + 1, 10))) # use positive sources for now

    az_src = slab.Trialsequence(conditions=sources[0], n_reps=n_trials)
    ele_src = slab.Trialsequence(conditions=sources[1], n_reps=n_trials)

    # stims = slab.Precomputed(lambda: slab.Binaural.pinknoise(samplerate=hrtf.samplerate, duration=0.2), n=30)
    sound = slab.Binaural.pinknoise(samplerate=48828.125000, duration=10.0) # generate stim to load into proc buffer

    # start loop over trials
    proceed = 'y'
    for az, ele in zip(az_src, ele_src):  # get sound source coordinates for next trial
        if proceed == 'y':
            target = np.array([az, ele]) # get target sound position from src list for next trial
            [index] = np.intersect1d(hrtf.cone_sources(az), hrtf.elevation_sources(ele))
            stim=hrtf.apply(index, sound)    # create stim with sound source specific hrtf function applied
            proc.WriteTagV('soundL', 0, stim.data[:, 0])  # laod stim into buffer (stereo)
            proc.WriteTagV('soundR', 0, stim.data[:, 1])
            proceed = play_trial(target, goal, s_range, proc, vid) # play n trials #todo fix n_trials
    return 0

def play_trial(target,goal,s_range,proc,vid=True):
    print('starting..')
    cam = freefield.camera.initialize_cameras('FLIR')    # init cam
    proc.Run() #start processor

    imin = 20  # minimal inter stim interval in ms(if pose and sound source match)
    imax = 700  # max ISI in ms
    isi_params = {'imin': imin, 'imax': imax, 'Irange': imax-imin,
                 'max_dst': la.norm(np.array([s_range, s_range])-np.array([-s_range, -s_range]))}
    isi, _, _, _, _ = isi_from_image(target, isi_params, cam)    # compare initial pose and calculate ISI, get image for video stream
    proc.SetTagVal('isi_in', isi)   # send initial isi to tdt processor

    proc.SoftTrg(1)     # buffer trigger (read and play stim)
    proc.SoftTrg(3)     # pulse train trigger #todo make better buffer loop

    end = False     # end condition

    while not end: # within trial loop
        isi, pose_az, pose_ele, info, image = isi_from_image(target, isi_params, cam) # compare pose and calculate ISI
        proc.SetTagVal('isi', isi) # set ISI in rcx file (pulse generator)

        # criteria to end experiment (pose matches sound source)
        matching_pose = target[0] - goal[0] <= pose_az <= target[0] + goal[0] and \
            target[1] - goal[0] <= pose_ele <= target[1] + goal[0]
        if matching_pose:         # if pose matches target window, start counting time until
            init_time = time.time()         # goal time is met or pose no longer matches
            while matching_pose:
                isi, pose_az, pose_ele, info, image = isi_from_image(target, isi_params, cam)  # compare pose and calculate ISI
                proc.SetTagVal('isi_in', isi) # set isi to pulse train tag
                matching_pose = target[0] - goal[0] <= pose_az <= target[0] + goal[0] and \
                                target[1] - goal[0] <= pose_ele <= target[1] + goal[0] # check if goal conditions are still met
                print('ON TARGET for %i sec \n sound source, az: %i  ele: %i \n head pose, az: %f  ele: %f    ISI: %fs' % (
                int(time.time()-init_time), target[0], target[1], pose_az, pose_ele, isi / 1000)) # for testing
                if time.time() > init_time + goal[1]:
                    end = True
                    proc.SoftTrg(2)
                    proc.Halt()
                    cam.halt()
                    break

        if vid:  # optional: show video with marker orientation
            image = vid_marker_orientation(image, [pose_az, pose_ele], info)  # draw marker orientation on frame
            cv2.putText(image, 'TARGET: azimuth: %i, elevation: %i' % (target[0], target[1]), (50, 50), # display target sound location
                        cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 0, 0), lineType=1,
                    thickness=1)
            cv2.imshow("Image", image)  # show the output frame
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):  # if the `q` key was pressed, break
                end = True
                proc.SoftTrg(2)
                proc.Halt()
                cam.halt()
                break

        print('sound source, az: %i  ele: %i \n head pose, az: %f  ele: %f    ISI: %fs, ' % (target[0], target[1], pose_az, pose_ele, isi/1000)) # for testing
    return input('Goal! Conintue? (y/n)')

def isi_from_image(target, isi_params, cam):
    diff, pose_az, pose_ele, info, image = compare_pose(target, cam)  # compare pose, get info(r/tvecs) and image
    if np.isnan(diff): # if no marker is detected
        isi = isi_params['imax']
    else:
        # scale ISI with deviation of pose from soundsource
        #isi = isi_params['imin'] + isi_params['Irange'] * (np.log(diff/isi_params['max_dst']+0.05)+3)/3
        isi = isi_params['imin'] + isi_params['Irange'] * (diff / isi_params['max_dst'])

    print (diff)
    return isi, pose_az, pose_ele, info, image

def compare_pose(target, cam): # compare headpose to sound source target, get euclidean distance
    #pose = cam.get_head_pose(aruco=True, convert=False) # working alternative

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
            az, ele, info = cam.model.pose_from_image_aruco(image) #grab info file from headpose.py
            pose[:, i_image, i_cam] = az, ele
    if convert:
        if not calibration:
            raise ValueError("Camera has to be calibrated to convert coordinates")
        for i_cam in range(pose.shape[2]):  # convert all images for each cam ...
            for i_angle, angle in enumerate(["azimuth", "elevation"]):  # ... and each angle
                a, b = cam.calibration[f"cam{i_cam}"][angle]["a"], cam.calibration[f"cam{i_cam}"][angle]["b"]
                pose[i_angle, :, i_cam] = a + b * pose[i_angle, :, i_cam]
    if average_axis is not None:
        pose = pose.mean(axis=average_axis) #todo fix this

    return la.norm(pose-target), pose[0], pose[1], info, image

def vid_marker_orientation(image, pose, info):
    marker_len = .05
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)
    (corners, ids, rejected) = cv2.aruco.detectMarkers(image,arucoDict)
    if len(corners) > 0:
        imaxis = cv2.aruco.drawDetectedMarkers(image.copy(), corners, ids)
        # info=[camera_matrix, dist_coeffs, rotation_vec, translation_vec]
        image = cv2.aruco.drawAxis(imaxis, info[0], info[1], info[2], info[3], marker_len)
        bottomLeftCornerOfText = (int(corners[0][0][2][0]), int(corners[0][0][2][1]))
        cv2.putText(image, 'azimuth: %f elevation: %f' % (pose[0], pose[1]), # display heade pose
                    bottomLeftCornerOfText, cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(225, 225, 225), lineType=1,
                    thickness=1)

    return(image)

if __name__ == "__main__":
    run_exp([5,3],30,5, vid=True)