"""
from freefield import devices


def test_devices():
    _devices = devices.Devices()
    for mode in ["play_rec", "play_birec", "loctest_freefield",
                 "loctest_headphones", "cam_calibration"]:
        _devices.initialize_default(mode)

    for n_samples in [1, 1000]:
        data = _devices.read("tag", n_samples, proc='RX8')
        assert len(data) == n_samples

    assert _devices.write("tag", 1, procs=['RX81', 'RX82']) == 1
"""
