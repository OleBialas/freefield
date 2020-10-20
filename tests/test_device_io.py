from freefield import devices
devices._win = False


def test_devices():
    _devices = devices.Devices()
    for mode in ["play_rec", "play_birec", "loctest_freefield",
                 "loctest_headphones", "cam_calibration"]:
        _devices.initialize_default(mode)
