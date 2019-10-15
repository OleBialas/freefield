import numpy as np
import time
import slab
from freefield_toolbox import setup, headpose
import os
cd = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

def localize_noise(setup="arc", n_trials=100, dur=1.0, outfile=None):

    setup.set_setup(setup) # first we need to pick the setup ("dome" or "arc")
    # then we have to initialize the devices by giving a .rcx file for each device:
    rx8_path=os.path.join(cd, "play_buffer_mono.rcx")
    rp2_path=os.path.join(cd, "button_response.rcx")
    setup.initialize_devices(rx8_path, rx8_path, rp2_path, ZBus=True)
    cam = headpose.camera() # create an object for handling the camera
    cam.start() # start image acquisition
    stim = slab.Sound.whitenoise(duration=dur) # generate the stimulus
    setup.set("playbuflen",stim.nsamples, "RX8s") # set the length of the buffer
    setup.set("stim",stim.data, "RX8s") # load stimulus into the buffer
    response=[] # create empty list to write the response into
    for i in range(n_trials): # start presentation
        # select a random speaker, then set the channel number of the analog
        # output to the channel to which the chosen speaker is attached:
        speaker_nr = np.random.randint(1,48)
        channel, proc = setup.get_speaker_from_number(speaker_nr)
        setup.set("chan",channel, proc)
        setup.trigger() # start the presentation
        setup.wait_to_finish_playing() #wait while sound is being displayed
        # now we wait for the participant to localize the sound:
        while not setup.get("response", proc="RP2"):
            time.sleep(0.01)
        #when the button has been pressed, we get the current head pose,
        #which is described by the two angles azimuth and elevation:
        ele,azi,_ = cam.get_pose()
        time.sleep(0.2) # wait 200 ms before the next trial starts
        response.append([speaker_nr, ele, azi])
    #save the response to a .txt file:
    if outfile is not None:
        np.writetxt(response, outfile)
