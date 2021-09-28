import freefield
import slab
import numpy as np
import random

# init cam
cam=freefield.camera.initialize_cameras('webcam')
#generate sound
noise = slab.Binaural.pinknoise(samplerate=44100, duration=0.5)
# set range of azimuthal positions
azs=np.arange(-45,46,10)
# select random location and apply itd and ild to noise
repeat = 'y'
while repeat == 'y':
    az = random.choice(azs)
    #noise = noise.at_azimuth(azimuth=az)
    noise = noise.itd(slab.Binaural.azimuth_to_itd(az))
    noise=noise.externalize()
    compare_pose = False
    while compare_pose == False:
        noise.play()
        pose = cam.get_head_pose(aruco=True, convert=False)
        if az-5 < pose[0] < az+5: # compare head pose with sound source location
            compare_pose==True
            repeat = input('Hit! Restart? (y/n)')
