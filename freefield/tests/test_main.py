from freefield import main, DIR
import numpy as np
import os
import unittest
import pandas as pd
import slab
from freefield.tests.test_camera import VirtualCam
main.initialize_setup(setup="dome", default_mode="play_rec", camera_type=None)
# TODO: test arc as well!
cam = VirtualCam()
cam.calibrate(pd.read_csv(DIR / "tests" / "coordinates.csv"), plot=False)
main.Cameras = cam


class TestMainMethods(unittest.TestCase):

    def test_wait(self):
        main.play_and_wait()
        main.wait_for_button()
        main.wait_to_finish_playing()

    def test_get_speaker(self):
        # get single speakers
        for i in range(47):
            speaker = main.get_speaker(index_number=i, coordinates=None)
            assert speaker.index_number.iloc[0] == i
        for azi, ele in zip(main.TABLE.azi, main.TABLE.ele):
            speaker = main.get_speaker(coordinates=[azi, ele])
            assert speaker.azi.iloc[0] == azi
            assert speaker.ele.iloc[0] == ele
        # get lists of speakers:
        speaker_list = [4, 16, 32, 45]
        speakers = main.get_speaker_list(speaker_list)
        assert len(speakers) == len(speaker_list)
        speaker_list = [(-52.5, 25), (-35, -12.5), (0, -12.5)]
        speakers = main.get_speaker_list(speaker_list)
        assert len(speakers) == len(speaker_list)

    def test_shift_setup(self):
        for _ in range(10):
            index_number = np.random.randint(0, 47)
            pre_shift = main.get_speaker(index_number=index_number)
            delta = (np.round(np.random.uniform(-10, 10), 2), np.round(np.random.uniform(-10, 10), 2))
            main.shift_setup(delta_azi=delta[0], delta_ele=delta[1])
            post_shift = main.get_speaker(index_number=index_number)
            assert (post_shift.azi.iloc[0] - pre_shift.azi.iloc[0]).round(2) == delta[0]
            assert (post_shift.ele.iloc[0] - pre_shift.ele.iloc[0]).round(2) == delta[1]
        main.initialize_setup(setup="dome", default_mode="play_rec", camera_type=None)

    def test_set_signal_and_speaker(self):
        # TODO: test applying calibration
        signals = [np.random.random(size=1000), slab.Sound(np.random.random(size=1000))]
        speakers = [np.random.randint(0, 47), [0, 0]]
        procs = ["RX81", "RX82"]
        for signal in signals:
            for proc in procs:
                for speaker in speakers:
                    main.set_signal_and_speaker(signal, speaker, proc)

    def test_get_recording_delay(self):
        delay = main.get_recording_delay()
        assert delay == 227
        delay = main.get_recording_delay(play_from="RX8", rec_from="RP2")
        assert delay == 316

    def test_check_pose(self):
        assert main.check_pose(var=100) is True
        assert main.check_pose(var=0) is False

    def test_calibrate_camera(self):
        targets = main.all_leds()
        coords = main.calibrate_camera(targets, n_reps=1, n_images=1)

    def test_localization_test_freefield(self):
        targets = main.TABLE.head()
        seq = main.localization_test_freefield(targets=targets, duration=.8, n_reps=1, n_images=5, visual=False)
        assert len(seq.trials) == len(seq.data)

    def test_localization_test_headphones(self):
        targets = main.TABLE.head()
        signals = [slab.Precomputed(lambda: slab.Binaural([slab.Sound.pinknoise(), slab.Sound.pinknoise()]),
                                    n=10) for i in range(len(targets))]
        seq = main.localization_test_headphones(targets=targets, signals=signals, n_reps=1, n_images=5, visual=False)
        assert len(seq.trials) == len(seq.data)

    def test_play_and_record(self):
        speaker_nr = 23
        signal = slab.Sound.whitenoise()
        main.initialize_setup(setup="dome", default_mode="play_rec", camera_type=None)
        rec = main.play_and_record(speaker_nr, signal, compensate_delay=True, apply_calibration=True)

    def test_level_equalization(self):
        signal = slab.Sound.chirp(duration=0.05, from_frequency=100, to_frequency=20000)
        speaker_list = main.TABLE
        target_speaker = 23
        db_thresh = 80
        lvls = main._level_equalization(signal, speaker_list, target_speaker, db_thresh)
        assert len(lvls) == len(speaker_list)
        assert lvls[23] == 1

    def test_frequency_equalization(self):
        signal = slab.Sound.chirp(duration=0.05, from_frequency=100, to_frequency=20000)
        speaker_list = main.TABLE
        target_speaker = 23
        db_thresh = 80
        bandwidth = 1 / 10
        low_cutoff = 200
        high_cutoff = 16000
        alpha = 1.0
        lvls = main._level_equalization(signal, speaker_list, target_speaker, db_thresh)
        filter_bank = main._frequency_equalization(signal, speaker_list, target_speaker, lvls, bandwidth,
                                                   low_cutoff, high_cutoff, alpha, db_thresh)

    def test_equalize_speakers(self):
        n_files = len(os.listdir(DIR / "data" / "log"))
        main.equalize_speakers(speakers="all", target_speaker=23, bandwidth=1 / 10, db_tresh=80,
                               low_cutoff=200, high_cutoff=16000, alpha=1.0, plot=False, test=True)
        assert main.CALIBRATIONFILE.exists()
        assert len(os.listdir(DIR / "data" / "log")) == n_files + 1  # log folder should be one element longer
        calibration = main.CALIBRATIONDICT
        main.initialize_setup(setup="dome", default_mode="play_rec", camera_type=None)
        assert self.assertAlmostEqual(calibration, main.CALIBRATIONDICT)

def test_check_equialization():
    signal = slab.Sound.whitenoise()
    main.check_equalization(signal, speakers="all", max_diff=5, db_thresh=80)
    pass
