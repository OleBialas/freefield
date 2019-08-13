"""
Take pictures (at least 10, default is 20) of a chess board (which must be asymetric), find the internal
corners and compute the camera matrix and distortion coefficients based onthe position differences
between the images. For a detailed description of the procedure see:
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
"""

from freefield_toolbox import setup, camera
from pathlib import Path
import time
import os
import numpy as np
_rcx_file = Path(__file__).resolve().parents[0]/Path("button_response.rcx")
setup.initialize_devices(RP2_file=_rcx_file)

def calibrate_camera(out_folder, n_images=20, image_names="calib_im", remove_images=True):
    print("image acquisition started, press the button to take a picture")
    all_image_names=[]
    # take a picture everytime the button is pressed until n pictures have beeen recorded:
    for i in range(n_images):
        fname=Path(out_folder+image_names+i+".jpg")
        all_image_names.append(fname)
        while not setup.get_variable(variable="response", proc="RP2"):
            time.sleep(0.01) # wait untill button is pressed
        camera.acquire_images(fname) # take picture
    #compute and save camera matrix and distortion coefficients:
    mtx, dist = camera.compute_coefficients(all_image_names, board_shape=(9,6))
    np.save(Path(out_folder)/"camera_matrix.npy", mtx)
    np.save(Path(out_folder)/"distortion_coefficients.npy", mtx)

    if remove_images: # delete the recorded images:
        for im in all_image_names:
            os.remove(im)

    setup.halt()
    camera.halt()
