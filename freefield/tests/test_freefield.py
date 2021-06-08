import numpy.random
from freefield import freefield, DIR
import numpy as np
import os
import slab
from freefield.tests.test_camera import VirtualCam
freefield.initialize_setup(setup="dome", default_mode="play_rec", camera_type=None)
freefield.Cameras = VirtualCam(n_cams=1)


def test_wait():
    freefield.play_and_wait()
    freefield.wait_for_button()
    freefield.wait_to_finish_playing()


def test_get_speaker():

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


def test_shift_setup():
    for _ in range(10):
        index_number = np.random.randint(0, 47)
        pre_shift = freefield.get_speaker(index_number=index_number)
        delta = (np.round(np.random.uniform(-10, 10), 2), np.round(np.random.uniform(-10, 10), 2))
        freefield.shift_setup(delta_azi=delta[0], delta_ele=delta[1])
        post_shift = freefield.get_speaker(index_number=index_number)
        assert (post_shift.azi.iloc[0] - pre_shift.azi.iloc[0]).round(2) == delta[0]
        assert (post_shift.ele.iloc[0] - pre_shift.ele.iloc[0]).round(2) == delta[1]
    freefield.initialize_setup(setup="dome", default_mode="play_rec", camera_type=None)

def test_set_signal_and_speaker():
    # TODO: test applying calibration
    signals = [np.random.random(size=1000), slab.Sound(np.random.random(size=1000))]
    speakers = [np.random.randint(0, 47), [0, 0]]
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

def test_calibrate_camera():
    targets = freefield.all_leds()
    coords = freefield.calibrate_camera(targets, n_reps=1, n_images=1)

def test_localization_test_freefield():
    targets = freefield.TABLE.head()
    seq = freefield.localization_test_freefield(targets=targets, duration=.8, n_reps=1, n_images=5, visual=False)
    assert len(seq.trials) == len(seq.data)

def test_localization_test_headphones():
    targets = freefield.TABLE.head()
    signals = [slab.Precomputed(lambda: slab.Binaural([slab.Sound.pinknoise(), slab.Sound.pinknoise()]),
                                n=10) for i in range(len(targets))]
    seq = freefield.localization_test_headphones(targets=targets, signals=signals, n_reps=1, n_images=5, visual=False)
    assert len(seq.trials) == len(seq.data)

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
