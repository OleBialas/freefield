import sys
sys.path.append("C:/Projects/freefield_toolbox/")
from freefield_toolbox import camera, setup
from pathlib import Path
import time
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
_location = Path(__file__).resolve().parents[0]
rp2_file = _location/Path("button_response.rcx")
rx8_file = _location/Path("to_bits.rcx")
setup.set_speaker_config("arc")
setup.initialize_devices(RX81_file=str(rx8_file), RP2_file=str(rp2_file), ZBus=True, cam=True)

def calibrate_camera(out_folder, n_images=20, image_names="calib_im", remove_images=True):
    """
    Take pictures (at least 10, default is 20) of a chess board (which must be asymetric), find the internal
    corners and compute the camera matrix and distortion coefficients based onthe position differences
    between the images. For a detailed description of the procedure see:
    https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
    """
    print("image acquisition started, press the button to take a picture")
    all_image_names=[]
    # take a picture everytime the button is pressed until n pictures have beeen recorded:
    for i in range(n_images):
        fname=out_folder+image_names+str(i)+".jpg"
        all_image_names.append(fname)
        while not setup.get_variable(variable="response", proc="RP2"):
            time.sleep(0.01) # wait untill button is pressed
        camera.acquire_images(fname) # take picture
        setup.trigger()
    #compute and save camera matrix and distortion coefficients:
    input("all images recorded! press a button to continue")
    mtx, dist = camera.compute_coefficients(all_image_names, board_shape=(9,6))
    np.save(Path(out_folder)/"camera_matrix.npy", mtx)
    np.save(Path(out_folder)/"distortion_coefficients.npy", dist)

    if remove_images: # delete the recorded images:
        for im in all_image_names:
            os.remove(im)

    setup.halt()
    #camera.halt()

def calibrate_headpose(n_repeat=3, image_names="calib_im", remove_images=True ):
    """"
    Makes LEDs light up at the given postions. Subject has to align their head
    with the lit LED and push the button so a picture is taken and the head pose is
    determined. Then we can determine the coefficients of the linear regression for
    led position vs measured head position
    """
    # 1st column: bitmask (how the led is adressed) 2nd column: position in degree
    _leds = np.array([[1, 0], [2, 25.68], [4, 47.08], [8, 64.20]])
    print("image acquisition started, press the button to take a picture")
    trials = np.tile(_leds[:,0], n_repeat)
    results=np.zeros([len(trials), 2])
    results[:,0] = trials
    for i, count in zip(trials,range(len(trials))):
        print("trial "+str(i))
        setup.set_variable(variable="bitval",value=int(i), proc="RX81")
        tic = time.time()
        while not setup.get_variable(variable="response", proc="RP2"):
            print(setup.get_variable(variable="response", proc="RP2"))
            time.sleep(0.1) # wait untill button is pressed
        x,y,z = camera.get_headpose(n=5)
        results[count, 1]=y

    # remove None elements from results:
    pos = np.where(np.isnan(results)==True)[0]
    results = np.delete(results, pos, axis=0)
    print(str(len(trials)-len(results))+" trials were deleted...")

    linear_regressor = LinearRegression()
    linear_regressor.fit(results[:,0].reshape(-1,1), results[:,1].reshape(-1,1))
    pred = linear_regressor.predict(results[:,0].reshape(-1,1))

    plt.scatter(results[:,0].reshape(-1,1), results[:,1].reshape(-1,1))
    plt.plot(results[:,0].reshape(-1,1), pred)
    plt.show()

    return response

if __name__ =="__main__":
    response = calibrate_headpose(n_repeat=5)
