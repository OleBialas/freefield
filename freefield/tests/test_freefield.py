import numpy.random
from freefield import freefield
import numpy as np
import tempfile
from pathlib import Path
import slab
from freefield.tests.test_camera import VirtualCam


def test_initialize():
    for default in ['play_rec', 'play_birec', 'loctest_freefield', 'loctest_headphones', 'cam_calibration']:
        setup = np.random.choice(["dome", "arc"])
        freefield.initialize(setup=setup, default=default)


def test_wait():
    freefield.wait_for_button()
    freefield.wait_to_finish_playing()


def test_pick_speakers():
    for setup in ["dome", "arc"]:
        freefield.initialize(setup=setup, default="play_rec", camera=None)
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
            picks = numpy.random.choice(freefield.SPEAKERS, n_picks, replace=False)
            speakers = freefield.pick_speakers(picks)
            assert all(speakers == picks)


def test_calibrate_camera():
    for _ in range(5):
        freefield.initialize(setup="dome", default="cam_calibration", camera=None)
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

"""
def test_localization_test():
    freefield.initialize(setup="dome", default="loctest_freefield", camera=None)
    freefield.CAMERAS = VirtualCam(n_cams=numpy.random.randint(1, 4))
    freefield.calibrate_camera(freefield.all_leds(), n_reps=1, n_images=1, show=False)

    # calibrate the speakers
    signal = slab.Sound.chirp(duration=1.0, samplerate=48828)
    fbank = slab.Filter.equalizing_filterbank(signal, signal)
    for i in range(47):
        freefield.pick_speakers(i)[0].level=1
        freefield.pick_speakers(i)[0].filter = fbank

    for _ in range(5):
        n_speakers = numpy.random.randint(2, 10)
        speakers = numpy.random.choice(freefield.SPEAKERS, n_speakers, replace=False)
        duration = numpy.random.uniform(0, 2)
        n_reps = numpy.random.randint(1, 5)
        n_images = numpy.random.randint(1, 5)
        seq = freefield.localization_test_freefield(speakers, duration, n_reps, n_images, visual=False)
        assert len(seq.data) == len(speakers) * n_reps
        signals = [slab.Binaural.whitenoise()] * n_speakers
        seq = freefield.localization_test_headphones(speakers, signals, n_reps, n_images, visual=False)
        assert len(seq.data) == len(speakers) * n_reps
        speakers = freefield.all_leds()
        seq = freefield.localization_test_freefield(speakers, duration, n_reps, n_images, visual=True)
        assert len(seq.data) == len(speakers) * n_reps
"""

def test_set_signal_and_speaker():
    signals = [np.random.random(size=1000), slab.Sound(np.random.random(size=1000))]
    speakers = range(47)
    for signal in signals:
        for speaker in speakers:
            freefield.set_signal_and_speaker(signal, speaker, equalize=False)


def test_get_recording_delay():
    delay = freefield.get_recording_delay()
    assert delay == 227
    delay = freefield.get_recording_delay(play_from="RX8", rec_from="RP2")
    assert delay == 316


def test_check_pose():
    assert freefield.check_pose(var=100) is True
    assert freefield.check_pose(var=0) is False


def test_play_and_record():
    sound = slab.Sound.whitenoise()
    freefield.initialize(setup="dome", default="play_rec", camera=None)
    for speaker in freefield.SPEAKERS:
        rec = freefield.play_and_record(speaker, sound, compensate_delay=True, equalize=False)
        assert rec.n_samples == sound.n_samples
        assert rec.n_channels == 1
    freefield.initialize(setup="dome", default="play_birec", camera=None)
    for speaker in freefield.SPEAKERS:
        rec = freefield.play_and_record(speaker, sound, compensate_delay=True, equalize=False)
        assert rec.n_samples == sound.n_samples
        assert rec.n_channels == 2


def test_equalizing():
    freefield.initialize(setup="dome", default="play_rec", camera=None)
    sound = slab.Sound.chirp(duration=0.05, from_frequency=100, to_frequency=20000)
    speakers = numpy.random.choice(freefield.SPEAKERS, numpy.random.randint(1, 47), replace=False)
    target_speaker = numpy.random.choice(freefield.SPEAKERS)
    levels = freefield._level_equalization(speakers, sound, target_speaker, threshold=80)
    assert len(levels) == len(speakers)
    levels = freefield._level_equalization(speakers, sound, target_speaker, threshold=100)
    assert all(levels == 1)
    filter_bank, _ = freefield._frequency_equalization(speakers, sound, target_speaker, levels, 1 / 8,
                                                       200, 20000, 1.0, 80)
    assert filter_bank.n_channels


def test_equalization_file():
    freefield.SETUP = "dome"
    freefield.read_speaker_table()
    assert all([s.filter is None for s in freefield.SPEAKERS])
    _, filename = tempfile.mkstemp()
    freefield.equalize_speakers(file_name=filename)
    assert Path(filename).exists()
    freefield.load_equalization(filename)
    assert all(isinstance(s.filter, slab.Filter) for s in freefield.SPEAKERS)
    assert all(isinstance(s.level, float) for s in freefield.SPEAKERS)


def test_apply_equalization():
    freefield.initialize(setup="dome", default="play_rec", camera=None)
    sound = slab.Sound.whitenoise()
    speaker = numpy.random.choice(freefield.SPEAKERS)
    numpy.testing.assert_raises(ValueError, freefield.apply_equalization, sound, speaker)
    _, filename = tempfile.mkstemp()
    freefield.equalize_speakers(file_name=filename)
    freefield.load_equalization(filename)
    equalized = freefield.apply_equalization(sound, speaker)
    assert isinstance(equalized, slab.Sound)


def test_test_equalization():
    freefield.initialize(setup="dome", default="play_rec", camera=None)
    _, filename = tempfile.mkstemp()
    freefield.equalize_speakers(file_name=filename)
    freefield.load_equalization(filename)
    raw, leve, full = freefield.test_equalization()
    freefield.spectral_range(raw)
