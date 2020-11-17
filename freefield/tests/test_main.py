from freefield import main, DIR
import numpy as np
import pandas as pd
import slab
from freefield.tests.test_camera import VirtualCam
main.initialize_setup(setup="dome", default_mode="play_rec", camera_type=None)
# TODO: test arc as well!
cam = VirtualCam()
cam.calibrate(pd.read_csv(DIR / "tests" / "coordinates.csv"), plot=False)
main.Cameras = cam


def test_wait():
    main.play_and_wait()
    main.wait_for_button()
    main.wait_to_finish_playing()


def test_get_speaker():
    # get single speakers
    for i in range(47):
        speaker = main.get_speaker(index_number=i, coordinates=None)
        assert speaker.index_number.iloc[0] == i
    for azi, ele in zip(main._table.azi, main._table.ele):
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


def test_shift_setup():
    for _ in range(10):
        index_number = np.random.randint(0, 47)
        pre_shift = main.get_speaker(index_number=index_number)
        delta = (np.round(np.random.uniform(-10, 10), 2), np.round(np.random.uniform(-10, 10), 2))
        main.shift_setup(delta=delta)
        post_shift = main.get_speaker(index_number=index_number)
        assert (post_shift.azi.iloc[0] - pre_shift.azi.iloc[0]).round(2) == delta[0]
        assert (post_shift.ele.iloc[0] - pre_shift.ele.iloc[0]).round(2) == delta[1]
    main.initialize_setup(setup="dome", default_mode="play_rec", camera_type=None)


def test_set_signal_and_speaker():
    # TODO: test applying calibration
    signals = [np.random.random(size=1000), slab.Sound(np.random.random(size=1000))]
    speakers = [np.random.randint(0, 47), [0, 0]]
    procs = ["RX81", "RX82"]
    for signal in signals:
        for proc in procs:
            for speaker in speakers:
                main.set_signal_and_speaker(signal, speaker, proc)


def test_get_recording_delay():
    delay = main.get_recording_delay()
    assert delay == 227
    delay = main.get_recording_delay(play_device="RX8", rec_device="RP2")
    assert delay == 316


def test_check_pose():
    assert main.check_pose(var=100) is True
    assert main.check_pose(var=0) is False


def test_calibrate_camera():
    targets = main.all_leds()
    coords = main.calibrate_camera(targets, n_reps=1, n_images=1)


def test_localization_test_freefield():
    speakers = main._table.head()
    main.localization_test_freefield(speakers=speakers, duration=.8, n_reps=1, n_images=5, visual=False)
    pass
