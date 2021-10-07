import freefield
import headpose
import slab
import numpy as np
import random
import time
import threading
from numpy import linalg as la

# goal = {'max_angle_dev':3, 'time_on_tar': 0.5} # end conditions: target window size (az/ele angles) + timer
# range = 30 # set range of horizontal and vertical speaker locations (eg 30: -30° to 30° in elevation and azimuth)
# n_trials = 4
# run_exp(goal,range,n_trials)

def run_exp(goal, range, n_trials):

    # get hrtf fnction
    hrtf = slab.HRTF.kemar()
    # generate random instances of pink noise for n_trials
    stims = slab.Precomputed(lambda: slab.Binaural.pinknoise(samplerate=hrtf.samplerate, duration=0.2), n=30)

    # generate sequence of sound source coordinates
    sources = np.stack((np.arange(-range, range + 1, 10), np.arange(-range, range + 1, 10)))
    az_src = slab.Trialsequence(conditions=sources[0], n_reps=n_trials)
    ele_src = slab.Trialsequence(conditions=sources[1], n_reps=n_trials)

    # start loop over trials
    repeat = 'y'
    while repeat == 'y':
        for az, ele in zip(az_src, ele_src):  # get sound source coordinates for next trial
            target = np.array([az, ele])
            repeat = play_trial(target, goal, range)
    return 0

def play_trial(target,goal,range):
    print('starting..')
    cam = freefield.camera.initialize_cameras('webcam')    # init cam
    end = False     # end condition
    imin = 0.1  # minimal inter stim interval (if pose and sound source match)
    imax = 1.5  # max ISI
    isi_params = {'imin': imin, 'imax': imax, 'Irange': imax-imin,
                 'max_dst': la.norm(np.array([range,range])-np.array([-range,-range]))}

    isi,_,_ = runtime_isi(target,isi_params,cam) # compare initial pose and calculate ISI

    send_isi()
    # todo: send isi to tdt processor to attenuate pulse train timing

    while not end: # main loop
        isi,pose_az,pose_ele = runtime_isi(target, isi_params, cam)
        send_isi()

        # criteria to end experiment (pose matches sound source)
        matching_pose = target[0] - goal['max_angle_dev'] <= pose_az <= target[0] + goal['max_angle_dev'] and \
            target[1] - goal['max_angle_dev'] <= pose_ele <= target[1] + goal['max_angle_dev']
        if matching_pose:         # if pose matches target window, start counting time and refreshing ISIs until
            init_time=time.time()         # goal time is met or pose no longer matches (in that case go back to main loop)
            while matching_pose:
                pose_az,pose_ele = runtime_isi(target, isi_params, cam)
                send_isi()
                matching_pose = target[0] - goal['max_angle_dev'] <= pose_az <= target[0] + goal['max_angle_dev'] and \
                                target[1] - goal['max_angle_dev'] <= pose_ele <= target[1] + goal['max_angle_dev']
                if time.time() > init_time + goal['time_on_tar']:
                    end = True
    return input('Goal! Conintue? (y/n)')

def runtime_ISI(target, isi_params, cam):
    diff, pose_az, pose_ele = compare_pose(target, cam)  # compare pose
    if np.isnan(diff): # if no marker is detected
        isi = isi_params['imax']
    else:
        # scale ISI with deviation of pose from soundsource
        isi = isi_params['imin'] + isi_params['Irange'] * (np.log(diff/isi_params['max_dst']+0.05)+3)/3
    return isi, pose_az, pose_ele

def compare_pose(target,cam): # compare headpose to sound source target, get euclidean distance
    pose = cam.get_head_pose(aruco=True, convert=False)
    if pose[1]<0: #TODO: find better way to do this in headpose.py
        pose[1] += 180
    else: pose[1] -= 180
    pose[1] = pose[1] * -1
    return la.norm(pose-target),  pose[0], pose[1]

# todo: optimize runtimes
# tim=(time.time())
# diff = compare_pose(az=az, ele=ele)
# print(time.time()-tim)