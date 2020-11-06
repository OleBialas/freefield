from freefield import devices


def test_devices():
    _devices = devices.Devices()
    for mode in ["play_rec", "play_birec", "loctest_freefield",
                 "loctest_headphones", "cam_calibration"]:
        _devices.initialize_default(mode)
    _devices._procs
    data = _devices.read("tag", 1, proc='RX81')
    assert data == 1
    data = _devices.read("tag", 1000, proc='RX81')
    assert len(data) == 1000
    assert _devices.write("tag", 1, procs=['RX81', 'RX82']) == 1
