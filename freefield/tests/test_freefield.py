import numpy.random
from freefield import freefield, DIR
import numpy as np
import os
import slab
from freefield.tests.test_camera import VirtualCam


def test_wait():
    freefield.play_and_wait()
    freefield.wait_for_button()
    freefield.wait_to_finish_playing()


def test_pick_speakers():
    for setup in ["dome", "arc"]:
        freefield.initialize_setup(setup=setup, default_mode="play_rec", camera_type=None)
        indices = [s.index for s in freefield.SPEAKERS]
        coordinates = [(s.azimuth, s.elevation) for s in freefield.SPEAKERS]
        for _ in range(100):
            n_picks = numpy.random.randint(0, 47)
            picks = numpy.random.choice(indices, n_picks, replace=False)
            speakers = freefield.pick_speakers(picks)
            assert len(speakers) == n_picks
            assert [s.index for s in speakers].sort() == picks.sort()
            idx = np.random.randint(47, size=2)
            idx.sort()
            picks = coordinates[idx[0]:idx[1]]
            n_picks = len(picks)
            speakers = freefield.pick_speakers(picks)
            assert len(speakers) == n_picks
            assert [(s.azimuth, s.elevation) for s in speakers].sort() == picks.sort()


def test_calibrate_camera():
    for _ in range(5):
        freefield.initialize_setup(setup="dome", default_mode="cam_calibration", camera_type=None)
        n_cams = numpy.random.randint(1, 4)
        freefield.CAMERAS = VirtualCam(n_cams=n_cams)
        n_reps, n_images = numpy.random.randint(1, 5), numpy.random.randint(1, 5)
        speakers = freefield.all_leds()
        freefield.calibrate_camera(speakers, n_reps, n_images)
        cams = freefield.CAMERAS.calibration.keys()
        assert len(cams) == n_cams
        speakers = freefield.pick_speakers([1, 2, 3, 4, 5])
        numpy.testing.assert_raises(ValueError, freefield.calibrate_camera, speakers, n_reps, n_images)
        freefield.calibrate_camera_no_visual(speakers, n_reps, n_images)


def test_localization_test():
    freefield.initialize_setup(setup="dome", default_mode="loctest_freefield", camera_type=None)
    freefield.CAMERAS = VirtualCam(n_cams=numpy.random.randint(1,4))
    freefield.calibrate_camera(freefield.all_leds(), n_reps=1, n_images=1)
    for _ in range(5):
        n_speakers = numpy.random.randint(2, 10)
        speakers = numpy.random.choice(freefield.SPEAKERS, n_speakers, replace=False)
        duration = numpy.random.uniform(0, 2)
        n_reps = numpy.random.randint(1, 5)
        n_images = numpy.random.randint(1, 5)
        seq = freefield.localization_test_freefield(speakers, duration, n_reps, n_images, visual=False)
        assert len(seq.data) == len(speakers)*n_reps
        signals = [slab.Binaural.whitenoise()]*n_speakers
        seq = freefield.localization_test_headphones(speakers, signals, n_reps, n_images, visual=False)
        assert len(seq.data) == len(speakers)*n_reps
        speakers = freefield.all_leds()
        seq = freefield.localization_test_freefield(speakers, duration, n_reps, n_images, visual=True)
        assert len(seq.data) == len(speakers)*n_reps


def test_set_signal_and_speaker():
    signals = [np.random.random(size=1000), slab.Sound(np.random.random(size=1000))]
    speakers = range(47)
    procs = ["RX81", "RX82"]
    for signal in signals:
        for proc in procs:
            for speaker in speakers:
                freefield.set_signal_and_speaker(signal, speaker, proc)


def test_get_recording_delay():
    delay = freefield.get_recording_delay()
    assert delay == 227
    delay = freefield.get_recording_delay(play_from="RX8", rec_from="RP2")
    assert delay == 316


def test_check_pose():
    assert freefield.check_pose(var=100) is True
    assert freefield.check_pose(var=0) is False


def test_play_and_record():
    speaker_nr = 23
    signal = slab.Sound.whitenoise()
    freefield.initialize_setup(setup="dome", default_mode="play_rec", camera_type=None)
    rec = freefield.play_and_record(speaker_nr, signal, compensate_delay=True, apply_calibration=True)

def test_level_equalization():
    signal = slab.Sound.chirp(duration=0.05, from_frequency=100, to_frequency=20000)
    speaker_list = freefield.TABLE
    target_speaker = 23
    db_thresh = 80
    lvls = freefield._level_equalization(signal, speaker_list, target_speaker, db_thresh)
    assert len(lvls) == len(speaker_list)
    assert lvls[23] == 1

def test_frequency_equalization():
    signal = slab.Sound.chirp(duration=0.05, from_frequency=100, to_frequency=20000)
    speaker_list = freefield.TABLE
    target_speaker = 23
    db_thresh = 80
    bandwidth = 1 / 10
    low_cutoff = 200
    high_cutoff = 16000
    alpha = 1.0
    lvls = freefield._level_equalization(signal, speaker_list, target_speaker, db_thresh)
    filter_bank = freefield._frequency_equalization(signal, speaker_list, target_speaker, lvls, bandwidth,
                                                    low_cutoff, high_cutoff, alpha, db_thresh)

def test_equalize_speakers():
    n_files = len(os.listdir(DIR / "data" / "log"))
    freefield.equalize_speakers(speakers="all", target_speaker=23, bandwidth=1 / 10, db_tresh=80,
                                low_cutoff=200, high_cutoff=16000, alpha=1.0, plot=False, test=True)
    assert freefield.EQUALIZATIONFILE.exists()
    assert len(os.listdir(DIR / "data" / "log")) == n_files + 1  # log folder should be one element longer
    calibration = freefield.EQUALIZATIONDICT
    freefield.initialize_setup(setup="dome", default_mode="play_rec", camera_type=None)


def test_check_equialization():
    signal = slab.Sound.whitenoise()
    freefield.check_equalization(signal, speakers="all", max_diff=5, db_thresh=80)
pass
