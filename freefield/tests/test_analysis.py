from freefield import freefield, DIR, analysis
from freefield.tests.test_camera import VirtualCam
import pandas as pd
# generate some data:
freefield.initialize_setup(setup="dome", default_mode="play_rec", camera_type=None)
targets = freefield.TABLE.sample(10)
cam = VirtualCam()
cam.calibrate(pd.read_csv(DIR / "tests" / "coordinates.csv"), plot=False)
freefield.CAMERAS = cam
sequence = freefield.localization_test_freefield(targets=targets, duration=.8, n_reps=5, n_images=5, visual=False)


def test_get_loctest_data():
    data = analysis.get_loctest_data(sequence)
    assert len(data) == len(sequence.trials) == len(sequence.data)