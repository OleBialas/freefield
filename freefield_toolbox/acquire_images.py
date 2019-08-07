"""
Take n pictures. The timing is given by the responsebox or a sound
"""

import os
import sys
sys.path.append("C:/Projects/freefield_toolbox")
from freefield_toolbox import setup, camera
import time
_location_ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


def image_series(n, folder, timing=None, filenames="image"):
    """
    n (int): number of images taken
    folder (str): path to the folder in which the images are stored
    time (float, int, None): time to wait between two images. If None (default)
    the timing is dictated by the reponse box:
    filenames (str or list of str): names under which the images are stored.
    If it's a list, it must be of length n. if its a str, a number from 0 to n
    is added to oit for each trial.

    """
    setup.initialize_setup("arc")

    if time is not None: # take images with a certain timing
        rcx_path = os.path.join(_location_,"sawtooth_mono.rcx")
        setup.initialize_devices(rcx_RX8_1=rcx_path)

        for i in range(n):
            if type(filenames)==str:
                file=folder+filenames+"%s.jpg" %(str(i))
            elif type(filenames)==list:
                file=folder+filenames[i]

            setup.trigger(trig='soft', proc="RX8_1")
            camera.acquire_images(file)
            time.sleep(timing)

    else: # take images when button is pressed
        rcx_path = os.path.join(_location_,"button_response.rcx")
        setup.initialize_devices(rcx_RP2=rcx_path)

        while not setup.get("response", proc="RP2"):
            time.sleep(0.01)

    #camera.deinit()

if __name__ == "__main__":
    time.sleep(10)
    n = 20
    folder = "C:/Projects/"
    filenames="chess"
    timing=2
    image_series(n, folder, timing)
    print("done!")
